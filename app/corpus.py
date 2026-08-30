"""冻结 Europe PMC XML 语料的段落化与轻量 BM25 检索。

本模块不联网；只读取 ``evidence_pool_manifest.json`` 中带 SHA-256 的 OA XML。
检索分数只用于候选召回，不是证据质量或科学可信度分数。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from app.schemas import CorpusPassage
from tools.literature.xml_anchor import EpmcXmlDocument, parse_epmc_fulltext_xml

_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+_.-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")


def normalize_text(value: str) -> str:
    return _SPACE_RE.sub(" ", value or "").strip()


def tokenize(value: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(value or "") if len(token) > 1 or token.isdigit()]


def _anchor_parts(text: str, *, exact_max: int = 320, context_max: int = 160) -> tuple[str, str, str]:
    """从完整段落提取段内 prefix/exact/postfix，而不是误用相邻段落作上下文。"""
    if len(text) <= exact_max:
        return "", text, ""
    start = (len(text) - exact_max) // 2
    end = start + exact_max
    return text[max(0, start - context_max) : start], text[start:end], text[end : end + context_max]


def _safe_parse_xml(path: Path) -> EpmcXmlDocument:
    """复用证据锚点模块的安全 JATS 解析器，避免应用层出现第二套安全口径。"""
    return parse_epmc_fulltext_xml(path.read_bytes())


class FrozenReviewCorpus:
    """从构建 manifest 装载段落，并在内存中进行确定性检索。"""

    def __init__(self, repo_root: str | Path, manifest_path: str | Path | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path is not None
            else self.repo_root / "eval" / "data" / "evidence_pool_manifest.json"
        )
        self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        reviews = self._manifest.get("reviews") or []
        self._review_by_pmid = {str(row["pmid"]): row for row in reviews}
        self._passages: list[CorpusPassage] | None = None

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

    @property
    def available_pmids(self) -> set[str]:
        return {
            str(row["pmid"])
            for row in self._manifest.get("fulltext") or []
            if row.get("path")
        }

    def load(self) -> list[CorpusPassage]:
        if self._passages is not None:
            return self._passages
        passages: list[CorpusPassage] = []
        for fulltext in self._manifest.get("fulltext") or []:
            rel_path = fulltext.get("path")
            if not rel_path:
                continue
            path = (self.repo_root / rel_path).resolve()
            if self.repo_root not in path.parents:
                raise ValueError(f"manifest 路径越界：{rel_path}")
            actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            expected_sha = str(fulltext.get("sha256") or "")
            if expected_sha and actual_sha != expected_sha:
                raise ValueError(f"XML SHA-256 不匹配：{path}")
            pmid = str(fulltext["pmid"])
            pmcid = str(fulltext.get("pmcid") or "")
            review = self._review_by_pmid.get(pmid) or {}
            passages.extend(
                self._extract_file(
                    path,
                    pmid=pmid,
                    pmcid=pmcid,
                    title=review.get("title"),
                    sha256=actual_sha,
                )
            )
        self._passages = passages
        return passages

    def _extract_file(
        self, path: Path, *, pmid: str, pmcid: str, title: str | None, sha256: str
    ) -> list[CorpusPassage]:
        document = _safe_parse_xml(path)
        raw_rows: list[tuple[str | None, str]] = []
        for paragraph in document.paragraphs:
            text = normalize_text(paragraph.text)
            if len(text) < 80:
                continue
            raw_rows.append((paragraph.location.section, text))

        output: list[CorpusPassage] = []
        for index, (section, text) in enumerate(raw_rows, start=1):
            prefix, anchor_exact, postfix = _anchor_parts(text)
            output.append(
                CorpusPassage(
                    passage_id=f"{pmcid or pmid}:p{index:04d}",
                    paper_id=f"PMID:{pmid}",
                    pmid=pmid,
                    pmcid=pmcid,
                    title=title,
                    section=section,
                    text=text,
                    anchor_exact=anchor_exact,
                    prefix=prefix,
                    postfix=postfix,
                    source_path=str(path.relative_to(self.repo_root)),
                    source_sha256=sha256,
                )
            )
        return output

    def search(
        self,
        queries: Iterable[str],
        *,
        source_pmids: Iterable[str] = (),
        top_k: int = 12,
    ) -> list[CorpusPassage]:
        """BM25 召回；PMID 过滤为空时搜索全部冻结 OA 综述。"""
        if top_k <= 0:
            return []
        allowed = {str(p).removeprefix("PMID:") for p in source_pmids if str(p).strip()}
        docs = [p for p in self.load() if not allowed or p.pmid in allowed]
        query_tokens = tokenize(" ".join(queries))
        if not docs or not query_tokens:
            return []
        tokenized = [tokenize(p.text) for p in docs]
        n_docs = len(docs)
        avgdl = sum(len(tokens) for tokens in tokenized) / max(1, n_docs)
        df: Counter[str] = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        qf = Counter(query_tokens)
        scored: list[tuple[float, CorpusPassage]] = []
        k1, b = 1.5, 0.75
        for passage, tokens in zip(docs, tokenized, strict=True):
            tf = Counter(tokens)
            score = 0.0
            for term, query_weight in qf.items():
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                denom = freq + k1 * (1 - b + b * len(tokens) / max(avgdl, 1.0))
                score += query_weight * idf * (freq * (k1 + 1)) / denom
            if score > 0:
                scored.append((score, passage))
        scored.sort(key=lambda item: (-item[0], item[1].passage_id))
        return [row.model_copy(update={"score": round(score, 6)}) for score, row in scored[:top_k]]
