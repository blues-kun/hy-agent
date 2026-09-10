# 工具 SFT · GRPO · RP-GRPO

训练小模型读取工具返回、选择下一步动作，并比较配对科研条件下的策略收益。主应用大模型、工具执行器与评分规则固定；仅更新工具策略的 LoRA 参数。

这里发布训练实现与配置。项目已提供[约 100 步 RP-GRPO / GRPO 阶段汇总](../../docs/stage-results.md)，后续继续训练并补充最终权重与完整记录。实际 RP 原文、实验测量与参考轨迹需按[数据契约](data/README.md)另行装配；下述命令为运行示例，不冒充阶段结果的原始运行配置。

## 对照设计

| 组别 | 配置 | 比较目标 |
| --- | --- | --- |
| B0 | 工具 SFT＋固定科学/取证规则，无 RL | 固定规则是否足够 |
| B1 | 同一 SFT＋GRPO，独立抽取任务 | 基础 RL 效果 |
| B2 | GRPO＋配对任务日程 | 配对组织的作用 |
| B3 | RP-GRPO，η=0.5＋同一配对日程 | 配对目标的增量 |
| LOO | η=0、固定尺度 leave-one-out | 区分目标与优势估计改动 |

B1–B3/LOO 使用同一合法任务池、SFT 起点、工具、奖励与预算。B2/B3/LOO 共享配对日程；B1 独立抽题。B0 单列规则干预，不作为未加规则 SFT 的准入成绩。方法与公式见 [RP_METHOD.md](RP_METHOD.md)。

## 1. 准备资源

在 `mito/` 目录执行，先安装[训练依赖](../README.md)，准备[本地基座](../models/README.md)与[RP 数据](data/README.md)。下面的 GPU 0 为示例，应选择实际空闲卡；每次使用新的输出目录。

工具 SFT 可以从基座初始化；如已有兼容的工具适配器，再显式增加 `--initial-adapter`。核验适配器与工具适配器分别管理。

## 2. 工具 SFT

先验证参考执行、哈希、分区和长度：

```bash
python -m rp_grpo.train_planner_sft --mode validate \
  --skip-overlength --max-length 8192 --max-reference-steps 6 \
  --validation-output runs/planner-validation-v1.json
```

再进行补训或初次训练：

```bash
python -m rp_grpo.train_planner_sft --mode train --execute --gpu 0 \
  --skip-overlength --max-length 8192 --max-reference-steps 6 \
  --max-steps 100 --max-seconds 7200 --grad-accum 8 \
  --learning-rate 0.00002 --output-dir runs/planner-sft-v1
```

每行只监督最后一个助手动作；真实工具返回与历史动作作为上下文。步数和长度检查按完整路径筛选，实际保留量以验证报告为准。

## 3. 实际 rollout 与 SFT 准入

无规则评测检查工具相关性、参数合法性、反馈后的继续与修复、任务完成和同任务多条有效路径。默认成功率门槛为 90%、参数合法率为 100%，还要求反馈对、错误恢复机会与任务族覆盖。具体判断由 `audit_sft.py` 的实际重放给出。

从已装配快照读取完整开发集任务数，再运行评测：

```bash
N_DEV_TASKS=$(python -c 'from rp_grpo.environment import load_tasks; print(len(load_tasks("dev")))')
python -m rp_grpo.train_policy --mode evaluate --arm B1 --execute --gpu 0 \
  --initial-adapter runs/planner-sft-v1/adapter --split dev \
  --max-eval-tasks "$N_DEV_TASKS" --eval-rollouts 4 --eval-sample \
  --max-turns 6 --max-new-tokens 512 --max-context-tokens 8192 \
  --max-seconds 7200 --output-dir runs/planner-gate-v1

python -m rp_grpo.audit_sft \
  --adapter runs/planner-sft-v1/adapter \
  --evaluation-report runs/planner-gate-v1/run_manifest.json \
  --readiness-output runs/planner-gate-v1/readiness.json \
  --output runs/planner-gate-v1/SFT_GATE.json
```

此处 `B1 + evaluate` 不执行 RL。退出码 2 表示准入未通过；应根据具体错误改进 SFT，再创建新的评测与 gate。正式 RL 校验 gate 所绑定的模型、适配器、数据、工具、提示词与报告哈希。

## 4. B0–B3 与 LOO

先查看单组启动计划，不加载 GPU：

```bash
python -m rp_grpo.run_experiment --arm B3 --mode train \
  --initial-adapter runs/planner-sft-v1/adapter \
  --sft-gate runs/planner-gate-v1/SFT_GATE.json --gpu 0
```

B0 仅评测同一 SFT 加规则；默认读取已装配快照中的完整 dev 分区：

```bash
python -m rp_grpo.run_experiment --arm B0 --mode evaluate --execute --gpu 0 \
  --initial-adapter runs/planner-sft-v1/adapter \
  --eval-rollouts 4 --eval-sample --max-seconds 7200 \
  --output-dir runs/B0-dev-v1
```

通过准入后，按同一配置运行强化学习组：

```bash
for arm in B1 B2 B3 LOO; do
  python -m rp_grpo.run_experiment --arm "$arm" --mode train --execute --gpu 0 \
    --initial-adapter runs/planner-sft-v1/adapter \
    --sft-gate runs/planner-gate-v1/SFT_GATE.json \
    --seed 20260909 --max-steps 12 --max-seconds 1800 \
    --output-dir "runs/${arm}-train-v1" || break
done
```

上述 12 步为工程小试配置。正式比较应事先确定共同预算与种子，记录生成 token、工具次数和墙钟时间。共享配置见 [experiment_config.json](experiment_config.json)。

## 5. 评测与结果

RL 产物路径是 `输出目录/adapter/policy/`。各 RL 组统一使用无规则 `--arm B1` 评测入口，只更换适配器，其他参数保持一致。例如：

```bash
python -m rp_grpo.train_policy --mode evaluate --arm B1 --execute --gpu 0 \
  --initial-adapter runs/B3-train-v1/adapter/policy --split dev \
  --max-eval-tasks "$N_DEV_TASKS" --eval-rollouts 4 --eval-sample \
  --max-turns 6 --max-new-tokens 512 --max-context-tokens 8192 \
  --max-seconds 7200 --output-dir runs/B3-dev-v1
```

完成各组相同任务与预算的评测后再汇总：

```bash
python -m rp_grpo.compare_runs \
  --run B0=runs/B0-dev-v1/run_manifest.json \
  --run B1=runs/B1-dev-v1/run_manifest.json \
  --run B2=runs/B2-dev-v1/run_manifest.json \
  --run B3=runs/B3-dev-v1/run_manifest.json \
  --run LOO=runs/LOO-dev-v1/run_manifest.json \
  --output runs/comparison-v1.json
```

主指标包括双侧成功率、真实任务完成率、错误支持或无效测量；同时报告提前停止、错误恢复、无效调用与成本。比较器独立重放实际记录，失败保留在分母中。统计单位是问题族或来源/实验组，不能将动作数或 G² 个配对分数当作独立样本。

当前实现为单机单卡 LoRA，checkpoint 支持权重续接，未实现完整优化器与随机状态恢复。线上系统按[接口契约](INTERFACES.md)适配；新模型或新工具版本需要重新验证。
