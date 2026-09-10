"""Portable launcher contracts. All children are mocked; never uses a GPU."""
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.posix


def load(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_code = Path(__file__).parent
if not (_code / "run_verifier_pipeline.py").is_file():
    _code = Path(__file__).parents[1] / "code"
_launcher = _code / "run_verifier_pipeline.py"
if not _launcher.is_file():
    _launcher = _code / "run_pipeline.py"
pipeline = load("isolated_verifier_pipeline", _launcher)
_verifier = _code / "train_verifier.py"
if not _verifier.is_file():
    _verifier = Path(__file__).parents[1] / "agent/training/qwen_llm/train_verifier.py"
verifier = load("isolated_pipeline_verifier_contract", _verifier)


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "portable package"
    for name in ("data/train.jsonl", "data/dev.jsonl", "code/train_verifier.py", "code/compare_verifier_finetune.py", "models/Qwen3-4B-Instruct-2507/config.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    return root


def cli(package, output, *extra):
    return ["--package-root", str(package), "--output-dir", str(output), "--gpu", "1", *extra]


def options(smoke=False):
    return SimpleNamespace(max_length=8192, max_new_tokens=1536, seed=20260908,
                           gpu=1, epochs=3, learning_rate=5e-5, smoke=smoke)


def test_default_plan_never_spawns_reads_gpu_or_creates_run(package, tmp_path, monkeypatch, capsys):
    output = tmp_path / "never-created"
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **k: pytest.fail("plan must not execute"))
    monkeypatch.setattr(pipeline.os, "killpg", lambda *a, **k: pytest.fail("plan must not signal"))
    before = sorted(str(p.relative_to(package)) for p in package.rglob("*"))
    assert pipeline.main(cli(package, output)) == 0
    assert not output.exists()
    assert sorted(str(p.relative_to(package)) for p in package.rglob("*")) == before
    text = capsys.readouterr().out
    assert "PLAN ONLY" in text and "no GPU use" in text
    for stage in ("validate:", "sft:", "base-dev:", "adapter-dev:", "compare:"):
        assert stage in text


def test_all_generated_commands_parse_against_actual_verifier_schema(package, tmp_path):
    jobs = pipeline.commands(options(), package, tmp_path / "run")
    assert [stage for stage, _ in jobs] == ["validate", "sft", "base-dev", "adapter-dev", "compare"]
    parsed = {}
    for name, command in jobs[:-1]:
        parsed[name] = verifier.build_parser().parse_args(command[2:])
        verifier.validate_args(parsed[name])
        assert "--allow-test" not in command
        assert "raw_candidates" not in " ".join(command)
    assert parsed["sft"].grad_accum == 8 and parsed["sft"].lora_r == 16
    assert parsed["base-dev"].adapter is None
    assert parsed["adapter-dev"].adapter == tmp_path / "run/sft/adapter"
    for field in ("model_path", "eval_data", "eval_split", "max_length", "max_new_tokens", "seed", "limit"):
        assert getattr(parsed["base-dev"], field) == getattr(parsed["adapter-dev"], field)


def test_smoke_is_two_steps_two_equal_dev_subsets_not_full_experiment(package, tmp_path):
    jobs = dict(pipeline.commands(options(smoke=True), package, tmp_path / "smoke"))
    assert verifier.build_parser().parse_args(jobs["sft"][2:]).max_steps == 2
    assert verifier.build_parser().parse_args(jobs["base-dev"][2:]).limit == 2
    assert verifier.build_parser().parse_args(jobs["adapter-dev"][2:]).limit == 2


@pytest.mark.parametrize("execute", [False, True])
def test_existing_output_rejected_before_any_launch(package, tmp_path, monkeypatch, execute):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original")
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **k: pytest.fail("existing run cannot launch"))
    with pytest.raises(SystemExit) as error:
        pipeline.main(cli(package, output, *(["--execute"] if execute else [])))
    assert error.value.code == 2
    assert marker.read_text() == "original" and len(list(output.iterdir())) == 1


def test_missing_package_file_fails_before_any_run(package, tmp_path):
    missing = package / "data/dev.jsonl"
    missing.unlink()
    output = tmp_path / "absent"
    with pytest.raises(SystemExit):
        pipeline.main(cli(package, output))
    assert not output.exists()


@pytest.mark.parametrize("flags", [["--max-hours", "9"], ["--max-hours", "nan"], ["--max-hours", "0"], ["--gpu", "-1"], ["--epochs", "4"]])
def test_invalid_budget_gpu_or_epochs_never_creates_run(package, tmp_path, flags):
    output = tmp_path / "absent"
    with pytest.raises(SystemExit):
        pipeline.main(cli(package, output, *flags))
    assert not output.exists()


class FakeChild:
    pid = 424242

    def __init__(self, command, *, stdout, stderr, start_new_session, fail_stage=None, incomplete=False):
        assert start_new_session is True  # killpg may only target this owned session.
        self.command = command
        self.returncode = None
        self.fail_stage = fail_stage
        self.incomplete = incomplete
        self.timeouts = []

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.timeouts.append(timeout)
        assert 0 < timeout <= 8 * 3600
        mode = self.command[self.command.index("--mode") + 1] if "--mode" in self.command else "compare"
        self.returncode = 7 if mode == self.fail_stage else 0
        if self.returncode == 0 and mode == "evaluate":
            output = Path(self.command[self.command.index("--output-dir") + 1])
            output.mkdir()
            (output / "evaluation.json").write_text(json.dumps({"all_selected_evaluated": not self.incomplete}))
        return self.returncode


def fake_launcher(monkeypatch, *, fail_stage=None, incomplete=False):
    children = []
    def popen(command, **kwargs):
        child = FakeChild(command, **kwargs, fail_stage=fail_stage, incomplete=incomplete)
        children.append(child)
        return child
    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    monkeypatch.setattr(pipeline.os, "killpg", lambda *a: pytest.fail("exited fake child must not be signalled"))
    return children


def test_mocked_execute_records_all_stages_with_bounded_budgets(package, tmp_path, monkeypatch):
    children = fake_launcher(monkeypatch)
    output = tmp_path / "mocked-run"
    assert pipeline.main(cli(package, output, "--execute", "--smoke", "--max-hours", "1")) == 0
    report = json.loads((output / "pipeline.json").read_text())
    assert report["status"] == "completed" and report["smoke_only"]
    assert len(report["stages"]) == len(children) == 5
    assert all(stage["status"] == "completed" for stage in report["stages"])
    for child in children:
        if "--max-seconds" in child.command:
            parsed = verifier.build_parser().parse_args(child.command[2:])
            verifier.validate_args(parsed)
            assert 120 < parsed.max_seconds <= 3600


def test_mocked_failure_never_runs_later_analysis_or_claims_completion(package, tmp_path, monkeypatch):
    children = fake_launcher(monkeypatch, fail_stage="train")
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="sft exited 7"):
        pipeline.main(cli(package, output, "--execute"))
    report = json.loads((output / "pipeline.json").read_text())
    assert report["status"] == "failed" and len(children) == 2
    assert report["stages"][-1]["exit_code"] == 7
    assert not (output / "base-dev").exists()


def test_incomplete_evaluation_cannot_progress_to_comparison(package, tmp_path, monkeypatch):
    children = fake_launcher(monkeypatch, incomplete=True)
    output = tmp_path / "incomplete"
    with pytest.raises(TimeoutError, match="evaluation incomplete"):
        pipeline.main(cli(package, output, "--execute"))
    report = json.loads((output / "pipeline.json").read_text())
    assert report["status"] == "failed" and len(children) == 3
    assert report["stages"][-1]["status"] == "budget_exhausted"
    assert not (output / "comparison.json").exists()


def test_small_remaining_budget_spawns_nothing(package, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **k: pytest.fail("budget cannot spawn"))
    output = tmp_path / "expired"
    with pytest.raises(TimeoutError, match="budget exhausted"):
        pipeline.main(cli(package, output, "--execute", "--max-hours", "0.01"))
    report = json.loads((output / "pipeline.json").read_text())
    assert report["status"] == "failed" and report["stages"] == []


def test_owned_child_cleanup_targets_only_the_created_process_group(monkeypatch):
    calls = []
    child = SimpleNamespace(pid=424242, poll=lambda: None, wait=lambda timeout: 0)
    monkeypatch.setattr(pipeline.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    pipeline.stop_owned_child(child)
    assert calls == [(424242, signal.SIGTERM)]
    calls.clear()
    child.poll = lambda: 0
    pipeline.stop_owned_child(child)
    assert calls == []


@pytest.mark.parametrize("vanished_on", [signal.SIGTERM, getattr(signal, "SIGKILL", 9)])
def test_child_exit_race_does_not_break_cleanup(monkeypatch, vanished_on):
    calls = []
    def kill(pid, sig):
        calls.append((pid, sig))
        if sig == vanished_on:
            raise ProcessLookupError("owned group already exited")
    def wait(timeout):
        raise pipeline.subprocess.TimeoutExpired("owned-child", timeout)
    child = SimpleNamespace(pid=424242, poll=lambda: None, wait=wait)
    monkeypatch.setattr(pipeline.os, "killpg", kill)
    pipeline.stop_owned_child(child)
    assert all(pid == 424242 for pid, _ in calls)
    assert calls[0][1] == signal.SIGTERM
    if vanished_on == signal.SIGKILL:
        assert calls[-1][1] == signal.SIGKILL


def test_launcher_sigterm_cleans_its_child_and_restores_handler(package, tmp_path, monkeypatch):
    original = object()
    handlers = {signal.SIGTERM: original}
    monkeypatch.setattr(pipeline.signal, "getsignal", lambda number: handlers[number])
    monkeypatch.setattr(pipeline.signal, "signal", lambda number, handler: handlers.update({number: handler}))
    kills, waits = [], []
    class InterruptedChild:
        pid = 424242
        def poll(self):
            return None
        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            return 0
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **kw: InterruptedChild())
    monkeypatch.setattr(pipeline.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt, match="received SIGTERM"):
        pipeline.main(cli(package, output, "--execute"))
    state = json.loads((output / "pipeline.json").read_text())
    assert state["status"] == state["stages"][-1]["status"] == "interrupted"
    assert kills == [(424242, signal.SIGTERM)]
    assert handlers[signal.SIGTERM] is original
    assert len(waits) == 2


def test_outer_wait_reserves_enough_time_for_owned_cleanup(package, tmp_path, monkeypatch):
    times = iter([0.0, 20.0, 30.0])
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: next(times))
    waits, kills = [], []
    class TimedOutChild:
        pid = 424242
        def poll(self):
            return None
        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise pipeline.subprocess.TimeoutExpired("owned-child", timeout)
            return 0
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **kw: TimedOutChild())
    monkeypatch.setattr(pipeline.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    output = tmp_path / "deadline"
    with pytest.raises(pipeline.subprocess.TimeoutExpired):
        pipeline.main(cli(package, output, "--execute", "--max-hours", "0.06"))
    remaining = int(0.06 * 3600 - 20.0)
    assert waits[0] <= remaining - 45
    assert waits[0] + 30 + 10 < remaining
    state = json.loads((output / "pipeline.json").read_text())
    assert state["status"] == state["stages"][-1]["status"] == "failed"
    assert kills == [(424242, signal.SIGTERM)]
