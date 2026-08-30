"""configs/judge_v0_1.yaml 的只读视图。

密钥纪律：API Key 只经环境变量（HY3_API_KEY）读取，本模块与配置文件都不保存
任何真实密钥；配置里只记录环境变量名。
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_JUDGE_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "judge_v0_1.yaml"


class JudgeConfigError(ValueError):
    """Judge 配置不自洽时抛出。"""


class JudgeConfig:
    """configs/judge_v0_1.yaml 的只读视图，带文件哈希供 run manifest 记录。"""

    def __init__(self, raw: Mapping[str, Any], source_path: Path | None = None, sha256: str = ""):
        self.raw = raw
        self.source_path = source_path
        self.sha256 = sha256
        self.version: str = str(raw["version"])
        self.frozen: bool = bool(raw.get("frozen", False))
        self.model: Mapping[str, Any] = raw["model"]
        self.transport: Mapping[str, Any] = raw["transport"]
        self.request: Mapping[str, Any] = raw["request"]
        self.structured_output: Mapping[str, Any] = raw["structured_output"]
        self.prompt_cache: Mapping[str, Any] = raw["prompt_cache"]
        self.self_consistency: Mapping[str, Any] = raw["self_consistency"]
        self._validate()

    def _validate(self) -> None:
        if float(self.transport["max_rps"]) > 1.0:
            # 模型级上限 RPM 60（实测）：超过 1 rps 必然触发限流。
            raise JudgeConfigError("transport.max_rps 不得超过 1.0（模型级 RPM 60）")
        channel = str(self.structured_output["channel"])
        if channel not in ("function_calling", "json_schema"):
            raise JudgeConfigError(f"未知结构化输出通道：{channel!r}")
        k = int(self.self_consistency["k"])
        min_votes = int(self.self_consistency["min_agreement_votes"])
        if not 1 <= min_votes <= k:
            raise JudgeConfigError(f"min_agreement_votes 必须在 1..k（{k}）之间，得到 {min_votes}")
        if float(self.self_consistency["temperature"]) <= 0:
            # 实测 temp=0 判定标签一致率 100%，采样无信息量。
            raise JudgeConfigError("self_consistency.temperature 必须 >0（temp=0 采样无信息量，实测）")

    # -- 环境变量解析（只读环境，不落盘） -------------------------------------

    def resolve_api_key(self) -> str:
        return os.environ.get(str(self.model["api_key_env"]), "")

    def resolve_base_url(self) -> str:
        return os.environ.get(str(self.model["base_url_env"]), "") or str(self.model["default_base_url"])

    def resolve_model(self) -> str:
        return os.environ.get(str(self.model["model_env"]), "") or str(self.model["default_model"])


def load_judge_config(path: str | Path | None = None) -> JudgeConfig:
    cfg_path = Path(path) if path is not None else DEFAULT_JUDGE_CONFIG_PATH
    text = cfg_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return JudgeConfig(yaml.safe_load(text), source_path=cfg_path, sha256=digest)


@lru_cache(maxsize=4)
def _cached(resolved: str) -> JudgeConfig:
    return load_judge_config(resolved)


def default_judge_config() -> JudgeConfig:
    """默认配置（带缓存）；测试中如需改配置请直接调用 load_judge_config。"""
    return _cached(str(DEFAULT_JUDGE_CONFIG_PATH))
