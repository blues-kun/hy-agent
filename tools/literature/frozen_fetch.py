"""Fetch the exact OA XML files referenced by a frozen evidence manifest.

Unlike ``build_gold_pool.py``, this module never rebuilds or mutates the
candidate pool.  It only materializes missing files whose bytes still match
the SHA-256 already frozen in ``evidence_pool_manifest.json``.  A changed
Europe PMC snapshot is reported as drift and is never silently accepted.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Protocol

from pydantic import Field

from evaluator.schemas import StrictModel


class FulltextClient(Protocol):
    def fetch_fulltext_xml(self, pmcid: str) -> tuple[str | None, str]: ...


class FrozenFetchItem(StrictModel):
    pmid: str
    pmcid: str
    path: str
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: str
    actual_sha256: str | None = None
    error: str | None = None


class FrozenFetchReport(StrictModel):
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected: int = Field(ge=0)
    ready: int = Field(ge=0)
    fetched: int = Field(ge=0)
    drifted: int = Field(ge=0)
    failed: int = Field(ge=0)
    complete: bool
    items: list[FrozenFetchItem]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_destination(repo_root: Path, rel_path: str) -> Path:
    if not rel_path or Path(rel_path).is_absolute():
        raise ValueError(f"manifest 全文路径必须是仓库相对路径：{rel_path!r}")
    destination = (repo_root / rel_path).resolve()
    if repo_root != destination and repo_root not in destination.parents:
        raise ValueError(f"manifest 全文路径越界：{rel_path!r}")
    return destination


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def fetch_frozen_corpus(
    *,
    manifest_path: str | Path,
    repo_root: str | Path,
    client: FulltextClient,
) -> FrozenFetchReport:
    """Materialize missing frozen XML files without changing the manifest."""

    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    rows = manifest.get("fulltext")
    if not isinstance(rows, list):
        raise ValueError("evidence manifest 缺少 fulltext 列表")

    items: list[FrozenFetchItem] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("path"):
            continue
        pmid = str(row.get("pmid") or "")
        pmcid = str(row.get("pmcid") or "")
        rel_path = str(row["path"])
        expected = str(row.get("sha256") or "").lower()
        if not pmid or not pmcid or len(expected) != 64:
            raise ValueError(f"冻结全文记录不完整：PMID={pmid!r}, PMCID={pmcid!r}")
        destination = _safe_destination(root, rel_path)

        if destination.is_file():
            actual = _sha256(destination.read_bytes())
            if actual == expected:
                items.append(
                    FrozenFetchItem(
                        pmid=pmid,
                        pmcid=pmcid,
                        path=rel_path,
                        expected_sha256=expected,
                        actual_sha256=actual,
                        status="ready_existing",
                    )
                )
            else:
                items.append(
                    FrozenFetchItem(
                        pmid=pmid,
                        pmcid=pmcid,
                        path=rel_path,
                        expected_sha256=expected,
                        actual_sha256=actual,
                        status="local_hash_mismatch",
                        error="本地文件与冻结 manifest 不一致；拒绝覆盖",
                    )
                )
            continue

        xml, error = client.fetch_fulltext_xml(pmcid)
        if xml is None:
            items.append(
                FrozenFetchItem(
                    pmid=pmid,
                    pmcid=pmcid,
                    path=rel_path,
                    expected_sha256=expected,
                    status="fetch_failed",
                    error=error or "Europe PMC 未返回 XML",
                )
            )
            continue
        data = xml.encode("utf-8")
        actual = _sha256(data)
        if actual != expected:
            items.append(
                FrozenFetchItem(
                    pmid=pmid,
                    pmcid=pmcid,
                    path=rel_path,
                    expected_sha256=expected,
                    actual_sha256=actual,
                    status="remote_snapshot_drift",
                    error="远端 XML 已变化；未写盘，须建立新 manifest 版本后再接受",
                )
            )
            continue
        _atomic_write(destination, data)
        items.append(
            FrozenFetchItem(
                pmid=pmid,
                pmcid=pmcid,
                path=rel_path,
                expected_sha256=expected,
                actual_sha256=actual,
                status="fetched_verified",
            )
        )

    ready = sum(item.status == "ready_existing" for item in items)
    fetched = sum(item.status == "fetched_verified" for item in items)
    drifted = sum(
        item.status in {"local_hash_mismatch", "remote_snapshot_drift"} for item in items
    )
    failed = sum(item.status == "fetch_failed" for item in items)
    return FrozenFetchReport(
        manifest_path=(
            str(manifest_file.relative_to(root))
            if manifest_file.is_relative_to(root)
            else manifest_file.name
        ),
        manifest_sha256=_sha256(manifest_bytes),
        expected=len(items),
        ready=ready,
        fetched=fetched,
        drifted=drifted,
        failed=failed,
        complete=ready + fetched == len(items),
        items=items,
    )


__all__ = ["FrozenFetchItem", "FrozenFetchReport", "fetch_frozen_corpus"]
