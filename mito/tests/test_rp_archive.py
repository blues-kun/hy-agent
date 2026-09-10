"""Small, CPU-only archive integrity tests; no real package replacement."""
import importlib.util
from pathlib import Path
import tarfile

import pytest


SPEC = importlib.util.spec_from_file_location("rp_archive_update", Path(__file__).resolve().parents[1] / "code/update_rp_archive.py")
archive_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive_mod)


def fixture_package(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    for name in archive_mod.REQUIRED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    (root / "PACKAGE.json").write_text('{"schema":"fixture"}')
    (root / "rp_grpo/SFT_GATE.json").write_text('{"passed":false,"allow_formal_rl":false}')
    (root / "SHA256SUMS").write_text("old manifest fixture\n")
    return root


def test_pack_preserves_old_archive_and_blocked_gate(tmp_path):
    root = fixture_package(tmp_path)
    archive = tmp_path / "bundle.tar.gz"
    original = b"original archive fixture"
    archive.write_bytes(original)
    result = archive_mod.seal_and_pack(root, archive, revision="test-rp")
    assert result["verified"] and not result["sft_gate_passed"]
    assert Path(result["prior_archive_backup"]).read_bytes() == original
    with tarfile.open(archive) as tar:
        assert "bundle/rp_grpo/SFT_GATE.json" in tar.getnames()
        assert all(m.isfile() for m in tar)


def test_missing_required_refuses_before_backup(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    archive = tmp_path / "bundle.tar.gz"
    archive.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="missing"):
        archive_mod.seal_and_pack(root, archive, revision="x")
    assert archive.read_bytes() == b"preserve"
    assert not list(tmp_path.glob("*.bak"))


def test_non_sibling_and_secret_paths_refused(tmp_path):
    root = fixture_package(tmp_path)
    with pytest.raises(ValueError, match="sibling"):
        archive_mod.seal_and_pack(root, root / "bundle.tar.gz", revision="x")
    (root / ".env").write_text("fixture, not a secret")
    with pytest.raises(ValueError, match="Secret"):
        archive_mod.package_files(root)


def test_symlink_refused_and_runs_not_packaged(tmp_path, make_symlink):
    root = fixture_package(tmp_path)
    (root / "runs").mkdir()
    (root / "runs/training.log").write_text("not bundled")
    assert not any("runs" in p.parts for p in archive_mod.package_files(root))
    make_symlink(root / "linked", root / "README.md")
    with pytest.raises(ValueError, match="symlink"):
        archive_mod.package_files(root)


def test_prior_checksum_mismatch_refused(tmp_path):
    root = fixture_package(tmp_path)
    archive = tmp_path / "bundle.tar.gz"
    archive.write_bytes(b"preserve")
    (tmp_path / "bundle.tar.gz.sha256").write_text("0" * 64 + "  bundle.tar.gz\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        archive_mod.seal_and_pack(root, archive, revision="x")
    assert archive.read_bytes() == b"preserve"


@pytest.mark.parametrize("suffix", [".x.verification.json", ".sha256.x.partial", ".x.partial"])
def test_preexisting_transaction_outputs_fail_before_any_change(tmp_path, suffix):
    root = fixture_package(tmp_path)
    archive = tmp_path / "bundle.tar.gz"
    archive.write_bytes(b"original")
    (tmp_path / (archive.name + suffix)).write_text("existing")
    with pytest.raises(ValueError, match="already exists"):
        archive_mod.seal_and_pack(root, archive, revision="x")
    assert archive.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.bak"))
    assert not (root / "audit/package-before-x").exists()


@pytest.mark.parametrize("fail_on", ["checksum", "receipt"])
def test_commit_failure_restores_original_archive_and_checksum(tmp_path, monkeypatch, fail_on):
    root = fixture_package(tmp_path)
    archive = tmp_path / "bundle.tar.gz"
    archive.write_bytes(b"original")
    checksum = tmp_path / "bundle.tar.gz.sha256"
    checksum.write_text(archive_mod.digest_file(archive) + "  bundle.tar.gz\n")
    original_checksum = checksum.read_text()
    real_replace = archive_mod.os.replace
    failed = []

    def replace(src, dst):
        target = checksum if fail_on == "checksum" else tmp_path / "bundle.tar.gz.x.verification.json"
        if Path(dst) == target and not failed:
            failed.append(True)
            raise OSError("injected commit failure")
        return real_replace(src, dst)

    monkeypatch.setattr(archive_mod.os, "replace", replace)
    with pytest.raises(RuntimeError, match="prior archive/checksum restored"):
        archive_mod.seal_and_pack(root, archive, revision="x")
    assert archive.read_bytes() == b"original"
    assert checksum.read_text() == original_checksum
    assert (tmp_path / "bundle.tar.gz.pre-x.bak").read_bytes() == b"original"
