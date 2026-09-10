"""Export reviewed verifier annotations without internal source directories.

Only metadata.original_record.file is reduced to its source basename. Prompts,
completions, labels, split assignments and original content hashes are preserved.
This is a publication transform, not a new annotation or approval step.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def public_record(row: dict) -> dict:
    result = deepcopy(row)
    source = result.get("metadata", {}).get("original_record", {})
    if isinstance(source.get("file"), str):
        source["file"] = source["file"].replace("\\", "/").rsplit("/", 1)[-1]
    # Refuse unexpected private fields instead of silently altering scientific data.
    payload = json.dumps(result, ensure_ascii=False)
    if re.search(r"/(?:home|storage|mnt)/|(?<![A-Za-z0-9])[A-Za-z]:[/\\\\]", payload):
        raise ValueError("Unexpected absolute path outside the source-file metadata")
    return result


def export(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if source == destination:
        raise ValueError("Source and public export must be separate directories")
    names = ("train.jsonl", "dev.jsonl", "FORMAT_CONTRACT.json", "MANIFEST.json")
    if any((destination / name).exists() for name in names):
        raise FileExistsError("Refusing to overwrite an existing export")
    records = {}
    buffers = {}
    sources = {}
    for split in ("train", "dev"):
        filename = f"{split}.jsonl"
        raw = (source / filename).read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if not rows or any(row["metadata"]["split"] != split for row in rows):
            raise ValueError(f"Invalid {split} partition")
        cleaned = [public_record(row) for row in rows]
        for before, after in zip(rows, cleaned):
            assert before["prompt"] == after["prompt"]
            assert before["completion"] == after["completion"]
        records[split] = cleaned
        buffers[filename] = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                               separators=(",", ":")) for row in cleaned) + "\n").encode("utf-8")
        sources[filename] = {"sha256": digest(raw), "bytes": len(raw)}
    for key in ("id", "task_id"):
        if {row[key] for row in records["train"]} & {row[key] for row in records["dev"]}:
            raise ValueError(f"Train/dev overlap in {key}")
    buffers["FORMAT_CONTRACT.json"] = (source / "FORMAT_CONTRACT.json").read_bytes()
    manifest = {
        "schema_version": "mito.public-annotations.v1",
        "data_role": "expert_evidence_summary_conditioned",
        "training_files": ["train.jsonl"],
        "evaluation_files": ["dev.jsonl"],
        "test_exported": False,
        "new_expert_approval": False,
        "publication_transform": {
            "changed_field": "metadata.original_record.file",
            "operation": "retain source basename; remove internal absolute directory",
            "prompt_completion_labels_preserved": True,
            "original_record_hashes_preserved": True,
        },
        "source_files": sources,
        "files": {name: {"sha256": digest(data), "bytes": len(data)} for name, data in buffers.items()},
        "counts": {},
    }
    for split, rows in records.items():
        counts = Counter(json.loads(row["completion"][0]["content"])["labels"]["evidence_support"] for row in rows)
        manifest["counts"][split] = {
            "records": len(rows), "independent_tasks": len({r["task_id"] for r in rows}),
            "evidence_support": dict(sorted(counts.items())),
        }
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in buffers.items():
        (destination / name).write_bytes(data)
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export(args.source, args.output)
    print(json.dumps({"counts": manifest["counts"], "files": manifest["files"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
