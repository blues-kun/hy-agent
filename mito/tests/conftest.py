"""Run migration contracts without GPU access or original machine paths."""
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["VERIFIER_MODEL_PATH"] = str(ROOT / "models/Qwen3-4B-Instruct-2507")
os.environ["VERIFIER_SCRIPT"] = str(ROOT / "code/train_verifier.py")


def pytest_addoption(parser):
    parser.addoption("--integration-data-dir", type=Path, default=None,
                     help="Existing RP snapshot for optional integration tests; never downloaded.")
    parser.addoption("--integration-model-dir", type=Path, default=None,
                     help="Existing local tokenizer/model directory; never downloaded.")


def pytest_configure(config):
    for name, description in (
        ("integration", "requires separately supplied local artifacts"),
        ("requires_data", "requires the private RP snapshot"),
        ("requires_model", "requires the local Qwen tokenizer"),
        ("posix", "requires POSIX process-group and signal APIs"),
    ):
        config.addinivalue_line("markers", f"{name}: {description}")
    data_dir = config.getoption("--integration-data-dir")
    if data_dir is not None:
        if not (data_dir / "MANIFEST.json").is_file():
            raise pytest.UsageError("--integration-data-dir must contain MANIFEST.json")
        from rp_grpo import environment
        environment.DEFAULT_DATA = data_dir.resolve()
    model_dir = config.getoption("--integration-model-dir")
    if model_dir is not None:
        if not (model_dir / "tokenizer_config.json").is_file():
            raise pytest.UsageError("--integration-model-dir must contain tokenizer_config.json")
        os.environ["VERIFIER_MODEL_PATH"] = str(model_dir.resolve())


def pytest_collection_modifyitems(config, items):
    data_dir = config.getoption("--integration-data-dir") or ROOT / "rp_grpo/data"
    model_dir = Path(os.environ["VERIFIER_MODEL_PATH"])
    for item in items:
        needs_data = item.get_closest_marker("requires_data") or "actual_reference" in item.fixturenames
        needs_model = item.get_closest_marker("requires_model")
        if needs_data or needs_model:
            item.add_marker(pytest.mark.integration)
        if needs_data and not (data_dir / "MANIFEST.json").is_file():
            item.add_marker(pytest.mark.skip(reason="Private RP snapshot is not included; supply --integration-data-dir."))
        if needs_model and not (model_dir / "tokenizer_config.json").is_file():
            item.add_marker(pytest.mark.skip(reason="Local Qwen tokenizer is not included; supply --integration-model-dir."))
        if item.get_closest_marker("posix") and os.name != "posix":
            item.add_marker(pytest.mark.skip(reason="POSIX process-group contracts run on Linux; unavailable on Windows."))


@pytest.fixture
def make_symlink():
    def create(link, target):
        try:
            link.symlink_to(target)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                pytest.skip("Windows symlink privilege is unavailable; this contract runs on Linux.")
            raise
    return create
