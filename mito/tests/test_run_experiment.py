from argparse import Namespace
import json
from pathlib import Path

import pytest

from rp_grpo.run_experiment import ROOT, command_for
from rp_grpo import run_experiment


@pytest.fixture(autouse=True)
def synthetic_experiment_snapshot(tmp_path, monkeypatch):
    """Only the scheduling contract is exercised; no actual training tasks."""
    root = tmp_path / "synthetic-experiment"
    folder = root / "rp_grpo"
    (folder / "data").mkdir(parents=True)
    (folder / "experiment_config.json").write_bytes((ROOT / "rp_grpo/experiment_config.json").read_bytes())
    tasks = [{"id": "synthetic-train", "split": "train"},
             {"id": "synthetic-dev-a", "split": "dev"},
             {"id": "synthetic-dev-b", "split": "dev"}]
    (folder / "data/tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tasks), encoding="utf-8")
    monkeypatch.setattr(run_experiment, "ROOT", root)


def args(**kwargs):
    defaults = dict(arm="B3", mode="train", output_dir=None, seed=20260909,
                    model_path=None, initial_adapter=None, sft_gate=None, max_steps=12,
                    max_seconds=1800, gpu=None, execute=False, max_eval_tasks=None,
                    eval_rollouts=1, eval_sample=False)
    return Namespace(**(defaults | kwargs))


def test_plan_does_not_execute_and_b0_is_not_training():
    assert "--execute" not in command_for(args())
    with pytest.raises(ValueError, match="evaluation-only"):
        command_for(args(arm="B0"))


def test_relative_paths_resolve_before_subprocess_changes_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    command = command_for(args(model_path=Path("model"), initial_adapter=Path("adapter"),
                               sft_gate=Path("gate.json"), output_dir=Path("out")))
    for flag, name in (("--model-path", "model"), ("--initial-adapter", "adapter"),
                       ("--sft-gate", "gate.json"), ("--output-dir", "out")):
        assert command[command.index(flag) + 1] == str(tmp_path / name)


def test_full_dev_default_and_explicit_sampling():
    command = command_for(args(mode="evaluate", eval_sample=True, eval_rollouts=4))
    assert command[command.index("--max-eval-tasks") + 1] == "2"
    assert command[command.index("--eval-rollouts") + 1] == "4"
    assert "--eval-sample" in command


def test_loo_eta_zero_and_no_silent_rl_override():
    command = command_for(args(arm="LOO", execute=True, gpu=2))
    assert command[command.index("--eta") + 1] == "0.0"
    assert "--engineering-smoke" not in command
    assert command[command.index("--gpu") + 1] == "2"
