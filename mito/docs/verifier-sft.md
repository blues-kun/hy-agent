# 证据核验 SFT

使用 Qwen3-4B-Instruct-2507 的 BF16 LoRA，学习给定证据摘要与实验条件下的主张核验。输出为 `labels`、`rationale`、`corrected_text`，与工具策略 SFT 独立训练。

## 准备与校验

先按[环境说明](../README.md)安装依赖，并将基座放到 `models/Qwen3-4B-Instruct-2507/`。以下命令在 `mito/` 执行。

```bash
python code/train_verifier.py --mode validate \
  --train-data data/train.jsonl --eval-data data/dev.jsonl \
  --model-path models/Qwen3-4B-Instruct-2507 \
  --max-length 8192 --max-new-tokens 1536
```

校验读取本地 tokenizer、数据结构、分区与长度，不加载 GPU 模型。训练仅监督 assistant 输出；问题、证据、元数据及 padding 不计入目标损失。超长输入会拒绝，不自动截断证据。

## 训练

示例 GPU 编号为 0，运行时选择实际空闲卡；输出目录必须是新目录。

```bash
python code/train_verifier.py --mode train \
  --train-data data/train.jsonl --eval-data data/dev.jsonl \
  --model-path models/Qwen3-4B-Instruct-2507 \
  --gpu 0 --output-dir runs/verifier-sft-v1 \
  --epochs 3 --learning-rate 5e-5 --batch-size 1 --grad-accum 8 \
  --lora-r 16 --max-length 8192 --max-new-tokens 1536 \
  --max-seconds 7200
```

该命令会开始训练。墙钟参数是运行预算，实际完成步数、损失与中断状态以运行清单为准。核验入口不使用工具训练入口的 `--execute` 开关。

## 同条件比较

```bash
python code/train_verifier.py --mode evaluate \
  --eval-data data/dev.jsonl --model-path models/Qwen3-4B-Instruct-2507 \
  --gpu 0 --output-dir runs/verifier-base-dev-v1 \
  --max-length 8192 --max-new-tokens 1536

python code/train_verifier.py --mode evaluate \
  --eval-data data/dev.jsonl --model-path models/Qwen3-4B-Instruct-2507 \
  --adapter runs/verifier-sft-v1/adapter \
  --gpu 0 --output-dir runs/verifier-adapter-dev-v1 \
  --max-length 8192 --max-new-tokens 1536

python code/compare_verifier_finetune.py \
  --base runs/verifier-base-dev-v1/predictions.jsonl \
  --finetuned runs/verifier-adapter-dev-v1/predictions.jsonl \
  --train data/train.jsonl --out runs/verifier-comparison-v1.json
```

比较同一输入与证据下的逐类指标、误支持、过度保留判断、结构有效率与截断情况。验证集没有 `mixed`，因此该类的缺失应保留在报告中；解释与改写的科学质量需单独评估。

本实现保留运行和 checkpoint 产物，但不提供完整 optimizer/RNG 断点恢复契约。最终发布时补充基座版本、adapter 哈希、训练配置与对应评测记录。
