"""Deterministic retrieval used only by the A/B/C/D Pilot experiment.

Arm B is a sparse TF-IDF vector baseline.  It is intentionally not described
as dense or embedding retrieval.  Arm C starts from the same TF-IDF scores and
adds a small evidence graph built exclusively from frozen passage text and
metadata:

* adjacent passages in the same paper;
* passages sharing at least two uncommon lexical terms.

No expert/gold field is accepted by either constructor, which makes label
leakage structurally impossible in this layer.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from app.schemas import CorpusPassage


SPARSE_TFIDF_METHOD = "sparse_tfidf_fulltext_vector_v1"
FROZEN_GRAPH_METHOD = "sparse_tfidf_plus_frozen_evidence_graph_v1"

_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+_.-]*|\d+(?:\.\d+)?")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "been",
    "between",
    "cells",
    "could",
    "from",
    "have",
    "into",
    "more",
    "other",
    "results",
    "showed",
    "study",
    "that",
    "their",
    "these",
    "this",
    "using",
    "were",
    "which",
    "with",
}


def sparse_tokens(value: str) -> list[str]:
    """Tokenize English biomedical text and Chinese queries deterministically."""

    tokens = [token.casefold() for token in _LATIN_TOKEN.findall(value or "")]
    for run in _CJK_RUN.findall(value or ""):
        # Frozen JATS is mostly English, while Pilot questions are Chinese.
        # Keeping characters and bigrams avoids an empty query when no Latin
        # abbreviation is present; it is not word segmentation.
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return [token for token in tokens if token]


@dataclass(frozen=True)
class RetrievalResult:
    method: str
    passages: list[CorpusPassage]
    candidate_count: int
    seed_passage_ids: list[str]
    expanded_candidate_count: int
    graph_node_count: int = 0
    graph_edge_count: int = 0
    graph_adjacency_edge_count: int = 0
    graph_lexical_edge_count: int = 0
    construction_source: str = "frozen_corpus_text_and_metadata_only"


class SparseTfidfIndex:
    """An in-memory L2-normalized sparse TF-IDF passage index."""

    def __init__(self, passages: Sequence[CorpusPassage]):
        ids = [passage.passage_id for passage in passages]
        if len(ids) != len(set(ids)):
            raise ValueError("TF-IDF passage_id 必须唯一")
        self.passages = list(passages)
        self.by_id = {passage.passage_id: passage for passage in self.passages}
        self._counts: dict[str, Counter[str]] = {}
        document_frequency: Counter[str] = Counter()
        for passage in self.passages:
            # Title and section are frozen metadata.  Repeating the title once
            # gives it modest weight without introducing an external model.
            source = " ".join(
                part
                for part in (passage.title or "", passage.title or "", passage.section or "", passage.text)
                if part
            )
            counts = Counter(sparse_tokens(source))
            self._counts[passage.passage_id] = counts
            document_frequency.update(counts)
        n_documents = max(1, len(self.passages))
        self.document_frequency = dict(document_frequency)
        self.idf = {
            term: math.log((1.0 + n_documents) / (1.0 + frequency)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self.vectors = {
            passage_id: self._normalized_vector(counts)
            for passage_id, counts in self._counts.items()
        }

    def _normalized_vector(self, counts: Counter[str]) -> dict[str, float]:
        weighted = {
            term: (1.0 + math.log(frequency)) * self.idf[term]
            for term, frequency in counts.items()
            if frequency > 0 and term in self.idf
        }
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        return {} if norm == 0 else {term: value / norm for term, value in weighted.items()}

    def query_vector(self, queries: Iterable[str]) -> dict[str, float]:
        counts = Counter(sparse_tokens(" ".join(str(query) for query in queries)))
        return self._normalized_vector(counts)

    def score_all(
        self,
        queries: Iterable[str],
        *,
        source_pmids: Iterable[str] = (),
    ) -> dict[str, float]:
        allowed = {
            str(value).removeprefix("PMID:")
            for value in source_pmids
            if str(value).strip()
        }
        query = self.query_vector(queries)
        if not query:
            return {}
        scores: dict[str, float] = {}
        for passage in self.passages:
            if allowed and passage.pmid not in allowed:
                continue
            vector = self.vectors[passage.passage_id]
            score = sum(value * vector.get(term, 0.0) for term, value in query.items())
            if score > 0:
                scores[passage.passage_id] = score
        return scores

    def search(
        self,
        queries: Iterable[str],
        *,
        source_pmids: Iterable[str] = (),
        top_k: int = 12,
    ) -> RetrievalResult:
        if top_k <= 0:
            return RetrievalResult(
                method=SPARSE_TFIDF_METHOD,
                passages=[],
                candidate_count=0,
                seed_passage_ids=[],
                expanded_candidate_count=0,
            )
        scores = self.score_all(queries, source_pmids=source_pmids)
        ranked = sorted(scores, key=lambda item: (-scores[item], item))
        selected = ranked[:top_k]
        passages = [
            self.by_id[passage_id].model_copy(update={"score": round(scores[passage_id], 8)})
            for passage_id in selected
        ]
        return RetrievalResult(
            method=SPARSE_TFIDF_METHOD,
            passages=passages,
            candidate_count=len(scores),
            seed_passage_ids=selected,
            expanded_candidate_count=0,
        )


class FrozenEvidenceGraphRetriever:
    """TF-IDF plus one-hop reranking/expansion over a frozen lexical graph."""

    def __init__(
        self,
        index: SparseTfidfIndex,
        *,
        max_lexical_document_frequency: int = 30,
        top_terms_per_passage: int = 18,
    ):
        self.index = index
        self.passages = index.passages
        self.by_id = index.by_id
        self.edges: dict[str, dict[str, float]] = defaultdict(dict)
        self._edge_types: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._build_adjacency_edges()
        self._build_lexical_edges(
            max_document_frequency=max_lexical_document_frequency,
            top_terms_per_passage=top_terms_per_passage,
        )

    @staticmethod
    def _edge_key(left: str, right: str) -> tuple[str, str]:
        return (left, right) if left < right else (right, left)

    def _connect(self, left: str, right: str, weight: float, edge_type: str) -> None:
        if left == right:
            return
        value = max(self.edges[left].get(right, 0.0), float(weight))
        self.edges[left][right] = value
        self.edges[right][left] = value
        self._edge_types[self._edge_key(left, right)].add(edge_type)

    def _build_adjacency_edges(self) -> None:
        by_paper: dict[str, list[CorpusPassage]] = defaultdict(list)
        for passage in self.passages:
            by_paper[passage.pmid].append(passage)
        position = {passage.passage_id: index for index, passage in enumerate(self.passages)}
        for rows in by_paper.values():
            rows.sort(key=lambda passage: position[passage.passage_id])
            for left, right in zip(rows, rows[1:]):
                self._connect(left.passage_id, right.passage_id, 1.0, "same_paper_adjacent")

    def _build_lexical_edges(
        self,
        *,
        max_document_frequency: int,
        top_terms_per_passage: int,
    ) -> None:
        inverted: dict[str, list[str]] = defaultdict(list)
        for passage in self.passages:
            counts = self.index._counts[passage.passage_id]
            candidates = [
                term
                for term in counts
                if len(term) >= 3
                and term not in _STOPWORDS
                and 2 <= self.index.document_frequency.get(term, 0) <= max_document_frequency
            ]
            candidates.sort(key=lambda term: (-self.index.idf.get(term, 0.0), term))
            for term in candidates[:top_terms_per_passage]:
                inverted[term].append(passage.passage_id)

        shared: Counter[tuple[str, str]] = Counter()
        for passage_ids in inverted.values():
            unique = sorted(set(passage_ids))
            for left_index, left in enumerate(unique):
                for right in unique[left_index + 1 :]:
                    shared[(left, right)] += 1
        for (left, right), overlap in shared.items():
            if overlap < 2:
                continue
            self._connect(left, right, min(0.9, 0.35 + 0.1 * overlap), "lexical_overlap")

    @property
    def edge_count(self) -> int:
        return len(self._edge_types)

    @property
    def adjacency_edge_count(self) -> int:
        return sum("same_paper_adjacent" in kinds for kinds in self._edge_types.values())

    @property
    def lexical_edge_count(self) -> int:
        return sum("lexical_overlap" in kinds for kinds in self._edge_types.values())

    def search(
        self,
        queries: Iterable[str],
        *,
        source_pmids: Iterable[str] = (),
        top_k: int = 12,
    ) -> RetrievalResult:
        if top_k <= 0:
            return RetrievalResult(
                method=FROZEN_GRAPH_METHOD,
                passages=[],
                candidate_count=0,
                seed_passage_ids=[],
                expanded_candidate_count=0,
                graph_node_count=len(self.passages),
                graph_edge_count=self.edge_count,
                graph_adjacency_edge_count=self.adjacency_edge_count,
                graph_lexical_edge_count=self.lexical_edge_count,
            )
        base_scores = self.index.score_all(queries, source_pmids=source_pmids)
        ranked_base = sorted(base_scores, key=lambda item: (-base_scores[item], item))
        seeds = ranked_base[:top_k]
        if not seeds:
            return RetrievalResult(
                method=FROZEN_GRAPH_METHOD,
                passages=[],
                candidate_count=0,
                seed_passage_ids=[],
                expanded_candidate_count=0,
                graph_node_count=len(self.passages),
                graph_edge_count=self.edge_count,
                graph_adjacency_edge_count=self.adjacency_edge_count,
                graph_lexical_edge_count=self.lexical_edge_count,
            )

        allowed = {
            str(value).removeprefix("PMID:")
            for value in source_pmids
            if str(value).strip()
        }
        candidates = set(seeds)
        for seed in seeds:
            candidates.update(
                neighbor
                for neighbor in self.edges.get(seed, {})
                if not allowed or self.by_id[neighbor].pmid in allowed
            )
        max_base = max(base_scores.values())
        normalized = {
            passage_id: score / max_base for passage_id, score in base_scores.items()
        }
        combined: dict[str, float] = {}
        for candidate in candidates:
            propagation = max(
                (
                    normalized.get(seed, 0.0) * self.edges.get(seed, {}).get(candidate, 0.0)
                    for seed in seeds
                ),
                default=0.0,
            )
            # Same budget as B: graph evidence replaces, rather than appends to,
            # the sparse-vector result.
            combined[candidate] = 0.7 * normalized.get(candidate, 0.0) + 0.3 * propagation
        selected = sorted(combined, key=lambda item: (-combined[item], item))[:top_k]
        passages = [
            self.by_id[passage_id].model_copy(update={"score": round(combined[passage_id], 8)})
            for passage_id in selected
        ]
        return RetrievalResult(
            method=FROZEN_GRAPH_METHOD,
            passages=passages,
            candidate_count=len(base_scores),
            seed_passage_ids=seeds,
            expanded_candidate_count=len(candidates - set(seeds)),
            graph_node_count=len(self.passages),
            graph_edge_count=self.edge_count,
            graph_adjacency_edge_count=self.adjacency_edge_count,
            graph_lexical_edge_count=self.lexical_edge_count,
        )
