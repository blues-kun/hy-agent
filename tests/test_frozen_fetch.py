from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.literature.frozen_fetch import fetch_frozen_corpus


class _Client:
    def __init__(self, values: dict[str, tuple[str | None, str]]):
        self.values = values
        self.calls: list[str] = []

    def fetch_fulltext_xml(self, pmcid: str) -> tuple[str | None, str]:
        self.calls.append(pmcid)
        return self.values[pmcid]


def _manifest(root: Path, rows: list[dict]) -> Path:
    path = root / "eval" / "data" / "evidence_pool_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"fulltext": rows}), encoding="utf-8")
    return path


def _row(xml: str, *, pmid: str = "1", pmcid: str = "PMC1") -> dict:
    return {
        "pmid": pmid,
        "pmcid": pmcid,
        "path": f"eval/data/corpus_raw/{pmcid}.xml",
        "sha256": hashlib.sha256(xml.encode()).hexdigest(),
    }


def test_fetches_only_bytes_matching_frozen_hash(tmp_path: Path):
    xml = "<article><body><p>frozen</p></body></article>"
    manifest = _manifest(tmp_path, [_row(xml)])
    report = fetch_frozen_corpus(
        manifest_path=manifest,
        repo_root=tmp_path,
        client=_Client({"PMC1": (xml, "")}),
    )
    assert report.complete and report.fetched == 1
    assert report.manifest_path == "eval/data/evidence_pool_manifest.json"
    assert (tmp_path / "eval/data/corpus_raw/PMC1.xml").read_text() == xml


def test_existing_matching_file_does_not_call_network(tmp_path: Path):
    xml = "<article><body><p>ready</p></body></article>"
    manifest = _manifest(tmp_path, [_row(xml)])
    destination = tmp_path / "eval/data/corpus_raw/PMC1.xml"
    destination.parent.mkdir(parents=True)
    destination.write_text(xml)
    client = _Client({})
    report = fetch_frozen_corpus(
        manifest_path=manifest, repo_root=tmp_path, client=client
    )
    assert report.complete and report.ready == 1
    assert client.calls == []


def test_remote_drift_is_reported_and_not_written(tmp_path: Path):
    manifest = _manifest(tmp_path, [_row("<article>expected</article>")])
    destination = tmp_path / "eval/data/corpus_raw/PMC1.xml"
    report = fetch_frozen_corpus(
        manifest_path=manifest,
        repo_root=tmp_path,
        client=_Client({"PMC1": ("<article>changed</article>", "")}),
    )
    assert not report.complete and report.drifted == 1
    assert not destination.exists()


def test_manifest_path_escape_is_rejected(tmp_path: Path):
    xml = "<article/>"
    row = _row(xml)
    row["path"] = "../escape.xml"
    manifest = _manifest(tmp_path, [row])
    with pytest.raises(ValueError, match="路径越界"):
        fetch_frozen_corpus(
            manifest_path=manifest,
            repo_root=tmp_path,
            client=_Client({"PMC1": (xml, "")}),
        )
