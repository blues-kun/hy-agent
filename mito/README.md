# Mito · 训练与评测

面向科研任务的证据核验与工具决策训练。基于 Qwen3-4B-Instruct-2507，提供独立的核验 SFT、工具 SFT，以及 GRPO / RP-GRPO 对照实现。

训练阶段，工具策略直接生成动作并在研究环境中执行；核验器独立学习主张检查。面向工作台接入，两个分支通过应用适配层封装为 Hy3 可调用的领域辅助工具，返回证据提示、条件约束与行动建议，由 Hy3 统一编排。训练动作接口与应用工具返回不是同一契约，详见[应用接入](rp_grpo/INTERFACES.md#应用接入)。

| 任务 | 学习目标 | 入口 |
| --- | --- | --- |
| 证据核验 SFT | 根据证据与实验条件判断主张、解释与改写 | [训练说明](docs/verifier-sft.md) |
| 工具 SFT | 读取真实返回，选择下一步工具与参数 | [执行指南](rp_grpo/README.md) |
| GRPO / RP-GRPO | 在配对科研条件下优化成功率与执行成本 | [方法定义](rp_grpo/RP_METHOD.md) |

本目录发布源码、测试、实验配置和已整理的[核验标注](data/README.md)。当前展示[前 200 步阶段结果与 SFT 损失](../docs/stage-results.md)，历史约 100 步汇总单列保留。原始 Qwen 底座与最终 RP-GRPO step200 LoRA 已整理为独立发布包，网盘下载链接待补，见[模型与权重说明](models/README.md)。训练持续推进，完整运行记录后续补充；旧适配器、RP 原文与测量快照另行准备。

知识库初轮适配、五位医学工作者邮件征询、专家修订 SFT 与工具策略 RL 的衔接，见[训练与评测说明](../docs/training-and-evaluation.md)。该页同时区分原文自监督、合成数据自训练和现有公开数据的逐条来源。

## 环境

训练环境为 Linux、Python 3.12 和支持 BF16 的 NVIDIA GPU。以下命令均在仓库的 `mito/` 目录执行。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.9.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
```

只运行公共 CPU 单测可安装 `requirements-test.txt`，然后执行：

```bash
python -m pip install -r requirements-test.txt
python -X utf8 -B -m pytest -p no:cacheprovider tests -q
```

缺少外置数据或本地模型时，相关集成测试会跳过；测试输出中的通过与跳过应分别记录。CPU 测试不启动训练。

已持有兼容快照与基座时，可在同一测试命令后添加 `--integration-data-dir /path/to/rp-data` 和 `--integration-model-dir /path/to/local-model`，执行对应的只读集成检查。

## 目录

```text
mito/
├── code/                    核验 SFT、比较与数据检查
├── data/                    公开核验标注、输出契约与清单
├── docs/                    核验训练说明
├── models/                  本地基座位置说明
├── rp_grpo/
│   ├── data/                RP 外置数据契约
│   ├── checkpoints/         适配器位置与发布说明
│   ├── environment.py       七个只读工具与执行环境
│   ├── train_planner_sft.py  工具 SFT
│   ├── train_policy.py       GRPO / RP-GRPO
│   └── experiment_config.json
└── tests/                   单元与集成测试
```

## 运行顺序

1. 按[模型说明](models/README.md)准备本地基座；核验任务可直接读取公开标注。
2. 工具训练需先装配[RP 数据快照与参考轨迹](rp_grpo/data/README.md)。
3. 完成工具 SFT，再使用无规则实际 rollout 生成准入报告。
4. 准入通过后，以同一 SFT 起点执行 B0–B3 与 LOO 对照。

所有运行产物写入新的 `runs/` 子目录。训练、准入记录和最终模型版本分别保存；不要把核验适配器当作工具策略起点。

→ [工具与接入契约](rp_grpo/INTERFACES.md) · [模型与权重](rp_grpo/checkpoints/README.md) · [返回项目首页](../README.md)
