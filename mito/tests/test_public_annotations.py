"""Public annotations retain exact targets and publish verifiable file hashes."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_public_annotations", ROOT / "code/export_public_annotations.py")
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def test_source_path_transform_preserves_training_content():
    row = {"prompt": [{"content": "question"}], "completion": [{"content": "labels"}],
           "metadata": {"original_record": {"file": "/home/private/claims.jsonl", "record_sha256": "abc"}}}
    result = EXPORTER.public_record(row)
    assert result["metadata"]["original_record"]["file"] == "claims.jsonl"
    assert row["metadata"]["original_record"]["file"] == "/home/private/claims.jsonl"
    assert result["prompt"] == row["prompt"]
    assert result["completion"] == row["completion"]
    assert result["metadata"]["original_record"]["record_sha256"] == "abc"


def test_unexpected_internal_path_is_not_silently_redacted():
    with pytest.raises(ValueError, match="Unexpected absolute path"):
        EXPORTER.public_record({"prompt": [{"content": "/storage/private/example"}]})


def test_source_urls_are_not_confused_with_windows_paths():
    row = {"prompt": [{"content": "https://doi.org/10.1234/example"}]}
    assert EXPORTER.public_record(row) == row


def test_public_manifest_and_partition_integrity():
    directory = ROOT / "data"
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, metadata in manifest["files"].items():
        content = (directory / name).read_bytes()
        assert len(content) == metadata["bytes"]
        assert hashlib.sha256(content).hexdigest() == metadata["sha256"]
    partitions = {}
    for split, expected in (("train", 443), ("dev", 51)):
        rows = [json.loads(line) for line in (directory / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == expected == manifest["counts"][split]["records"]
        assert all(row["metadata"]["split"] == split for row in rows)
        assert len({row["id"] for row in rows}) == expected
        assert all(EXPORTER.public_record(row) == row for row in rows)
        partitions[split] = rows
    for key in ("id", "task_id"):
        assert not {r[key] for r in partitions["train"]} & {r[key] for r in partitions["dev"]}
    assert manifest["test_exported"] is False
