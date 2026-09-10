"""Seal and atomically update this migration archive, retaining the prior archive.

This command packages files; it never trains, deploys, extracts an archive, or
removes previous results. Run only after all writers/tests have finished.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
from datetime import datetime, timezone


EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git", ".venv", "runs"}
REQUIRED = {"PACKAGE.json", "README.md", "rp_grpo/algorithms.py", "rp_grpo/train_policy.py",
            "rp_grpo/environment.py", "rp_grpo/SFT_GATE.json", "rp_grpo/README.md",
            "rp_grpo/RP_METHOD.md", "rp_grpo/INTERFACES.md", "rp_grpo/gate_validation.py",
            "rp_grpo/audit_sft.py", "rp_grpo/train_planner_sft.py", "rp_grpo/run_experiment.py",
            "rp_grpo/compare_runs.py", "rp_grpo/experiment_config.json",
            "rp_grpo/data/MANIFEST.json", "rp_grpo/data/tasks.jsonl", "rp_grpo/data/scorers.jsonl",
            "rp_grpo/data/passages.jsonl", "rp_grpo/data/views.jsonl", "rp_grpo/data/measurements.csv",
            "rp_grpo/data/REFERENCE_MANIFEST.json", "rp_grpo/data/reference_traces.jsonl",
            "rp_grpo/data/planner_sft.train.jsonl", "rp_grpo/data/planner_sft.dev.jsonl",
            "rp_grpo/checkpoints/tool_sft_v2/adapter_model.safetensors",
            "rp_grpo/checkpoints/tool_sft_v2/adapter_config.json"}


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_files(root: Path) -> list[Path]:
    rows = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Package symlink refused: {rel}")
        if not path.is_file() or rel.as_posix() == "SHA256SUMS":
            continue
        if path.name == ".env" or path.name.endswith((".key", ".pem", ".partial", ".tmp")):
            raise ValueError(f"Secret/unfinished artifact filename refused: {rel}")
        rows.append(path)
    rows.sort(key=lambda p: p.relative_to(root).as_posix())
    return rows


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def verify_tar(path: Path, prefix: str, expected: dict[str, str]) -> dict:
    seen = set()
    byte_count = 0
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or member.name in seen:
                raise ValueError(f"Nonregular/duplicate tar member: {member.name}")
            name = member.name.removeprefix(prefix + "/")
            if member.name != prefix + "/" + name or name not in expected:
                raise ValueError(f"Unexpected tar member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Missing tar stream")
            h = hashlib.sha256()
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
                byte_count += len(chunk)
            if h.hexdigest() != expected[name]:
                raise ValueError(f"Archive byte mismatch: {name}")
            seen.add(member.name)
    if len(seen) != len(expected):
        raise ValueError("Archive omitted required files")
    return {"regular_files": len(seen), "uncompressed_bytes": byte_count,
            "sha256": digest_file(path)}


def seal_and_pack(root: Path, archive: Path, *, revision: str) -> dict:
    root = root.resolve(strict=True)
    if not root.is_dir() or root == Path(root.anchor):
        raise ValueError("A concrete package directory is required")
    archive = archive.absolute()
    if archive.is_symlink() or archive.parent.resolve() != root.parent:
        raise ValueError("Archive must be a regular sibling of this package")
    if archive.name != root.name + ".tar.gz":
        raise ValueError("Archive must retain this package's exact original filename")
    if not revision or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in revision):
        raise ValueError("Invalid revision")
    missing = [name for name in sorted(REQUIRED) if not (root / name).is_file()]
    if missing:
        raise ValueError("Required extension files missing: " + ", ".join(missing))
    gate = json.loads((root / "rp_grpo/SFT_GATE.json").read_text())
    # A blocked gate is a valid and intentionally honest package state.
    if not isinstance(gate, dict) or "passed" not in gate:
        raise ValueError("SFT gate must contain an explicit passed decision")
    backup = archive.with_name(archive.name + ".pre-" + revision + ".bak")
    backup_checksum = backup.with_name(backup.name + ".sha256")
    previous = root / "audit" / ("package-before-" + revision)
    temporary = archive.with_name(archive.name + "." + revision + ".partial")
    checksum_path = archive.with_name(archive.name + ".sha256")
    checksum_temp = checksum_path.with_name(checksum_path.name + "." + revision + ".partial")
    receipt = archive.with_name(archive.name + "." + revision + ".verification.json")
    receipt_temp = receipt.with_name(receipt.name + ".partial")
    rollback = archive.with_name(archive.name + "." + revision + ".rollback.partial")
    checksum_rollback = checksum_path.with_name(checksum_path.name + "." + revision + ".rollback.partial")
    for target in (backup, backup_checksum, previous, temporary, checksum_temp, receipt, receipt_temp, rollback, checksum_rollback):
        if target.exists() or target.is_symlink():
            raise ValueError(f"Revision output already exists; inspect it or use a fresh revision: {target.name}")
    if checksum_path.is_symlink():
        raise ValueError("Original checksum must not be a symlink")
    package_files(root)  # reject unsafe files before modifying package metadata
    prior_checksum = checksum_path.read_text() if checksum_path.exists() else None
    if archive.exists():
        prior_hash = digest_file(archive)
        if prior_checksum is not None:
            if not prior_checksum.split() or prior_checksum.split()[0] != prior_hash:
                raise ValueError("Original archive checksum mismatch; refusing replacement")
        try:
            os.link(archive, backup)  # old inode survives atomic replacement
        except OSError:
            shutil.copy2(archive, backup)
        with backup_checksum.open("x") as handle:
            handle.write(prior_hash + "  " + backup.name + "\n")
    else:
        prior_hash = None

    previous.mkdir(parents=True, exist_ok=False)
    for name in ("SHA256SUMS", "PACKAGE.json"):
        if (root / name).is_file():
            shutil.copy2(root / name, previous / name)
    metadata = json.loads((root / "PACKAGE.json").read_text())
    metadata["rp_extension"] = {
        "schema": "mito.rp-migration-extension.v1", "revision": revision,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "prior_archive_sha256": prior_hash,
        "entrypoint": "rp_grpo/README.md", "sft_gate": "rp_grpo/SFT_GATE.json",
        "sft_gate_passed_at_seal": gate.get("passed") is True,
        "formal_rp_training_started": False, "production_deployment_changed": False,
        "main_arms": ["B0", "B1", "B2", "B3"],
        "attribution_control": "LOO_eta_0",
        "contains_historical_tool_sft_adapter": (root / "rp_grpo/checkpoints/tool_sft_v2/adapter_model.safetensors").is_file(),
        "scope": "Versioned offline bounded tools; see capability contract; not a full production deployment",
    }
    write_json(root / "PACKAGE.json", metadata)
    files = package_files(root)
    checksums = {p.relative_to(root).as_posix(): digest_file(p) for p in files}
    (root / "SHA256SUMS").write_text("".join(h + "  " + name + "\n" for name, h in checksums.items()))
    checksums["SHA256SUMS"] = digest_file(root / "SHA256SUMS")
    files.append(root / "SHA256SUMS")
    with temporary.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as tar:
                for path in files:
                    info = tar.gettarinfo(str(path), arcname=root.name + "/" + path.relative_to(root).as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as f:
                        tar.addfile(info, f)
        raw.flush()
        os.fsync(raw.fileno())
    verification = verify_tar(temporary, root.name, checksums)
    with checksum_temp.open("x") as f:
        f.write(verification["sha256"] + "  " + archive.name + "\n")
        f.flush()
        os.fsync(f.fileno())
    result = {"schema": "mito.rp-archive-verification.v1", "revision": revision,
              "archive": str(archive), "archive_bytes": temporary.stat().st_size,
              "prior_archive_backup": str(backup) if prior_hash else None,
              "prior_archive_sha256": prior_hash, "sft_gate_passed": gate.get("passed") is True,
              "verified": True, **verification}
    with receipt_temp.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    # Prepare every artifact before the short commit phase. Multiple directory
    # entries are not one atomic transaction: on failure restore the old archive
    # and checksum, retaining both the backup and staged artifacts for inspection.
    archive_committed = checksum_committed = False
    try:
        os.replace(temporary, archive)
        archive_committed = True
        os.replace(checksum_temp, checksum_path)
        checksum_committed = True
        os.replace(receipt_temp, receipt)
    except BaseException as exc:
        if archive_committed and prior_hash:
            try:
                os.link(backup, rollback)
            except OSError:
                shutil.copy2(backup, rollback)
            os.replace(rollback, archive)
        elif archive_committed:
            os.replace(archive, temporary)
        if checksum_committed:
            if prior_checksum is not None:
                with checksum_rollback.open("x") as f:
                    f.write(prior_checksum)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(checksum_rollback, checksum_path)
            else:
                os.replace(checksum_path, checksum_temp)
        raise RuntimeError("Archive commit failed; prior archive/checksum restored when present. Inspect staged files; use a fresh revision.") from exc
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.package_root.resolve(strict=True)
    archive = args.archive or root.with_name(root.name + ".tar.gz")
    if not args.execute:
        print(json.dumps({"action": "plan_only", "package": str(root), "archive": str(archive),
                          "revision": args.revision, "requires_quiescent_writers": True}))
        return
    print(json.dumps(seal_and_pack(root, archive, revision=args.revision), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
