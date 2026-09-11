# 模型权重与下载

权重不进入 Git 仓库。公开推理只需要 **原始 Qwen 底座 + 最终 policy LoRA** 两份材料，不再叠加旧 SFT 或 reference 适配器。

| 下载材料 | 内容 | 大小 | 地址 |
| :--- | :--- | ---: | :--- |
| 原始 Qwen3-4B-Instruct-2507 | 3 个分片、tokenizer、模型配置与原始许可证 | 约 8.06 GB | [Qwen 官方模型](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)；网盘镜像链接待补 |
| MITO RP-GRPO · step 200 | 最终 policy LoRA、适配器配置、模型卡与 SHA-256 | 约 132.2 MB | 独立发布包已整理，网盘链接待补 |

200 表示 **200 次 RL 采样循环和参数更新**，不是 200 个 epoch。最终策略以未合并的 PEFT LoRA 保存；不能脱离底座单独作为完整 4B 模型加载。格式说明见 [PEFT 官方文档](https://huggingface.co/docs/peft/v0.20.0/en/developer_guides/checkpoint)。

## 推理目录

将两个发布包解压到同一目录：

```text
weights/
├── Qwen3-4B-Instruct-2507/
│   ├── config.json
│   ├── model.safetensors.index.json
│   ├── model-*.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── LICENSE
└── MITO-RP-GRPO-step200-LoRA/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── README.md
    ├── release_manifest.json
    └── SHA256SUMS
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_path = "./weights/Qwen3-4B-Instruct-2507"
adapter_path = "./weights/MITO-RP-GRPO-step200-LoRA"
tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
base = AutoModelForCausalLM.from_pretrained(
    base_path, dtype="auto", device_map="cpu", local_files_only=True,
)
model = PeftModel.from_pretrained(
    base, adapter_path, local_files_only=True, is_trainable=False,
)
model.eval()
```

这段示例说明加载关系，未在本轮执行完整模型推理。依赖版本见[训练依赖](../requirements.txt)。实际工具任务还需要项目的提示格式、工具定义和执行环境。轻量发布包不含内部数据，也不用于恢复优化器与随机状态。

## 文件检查与标签更正

- 源复制清单的 20 个文件均通过 SHA-256 检查。底座的三个分片还与 Qwen 官方仓库公布的哈希一致。
- 底座 398 个张量、最终 LoRA 504 个张量的结构已核对；所有权重文件的数值扫描未发现 NaN / Inf。
- 源目录名称与旧检查点标签曾不一致。**项目负责人确认该发布为 RP-GRPO**；发布清单记录更正依据与旧标签，权重张量保持原样。文件哈希证明完整性，不代替训练方法的独立复现。
- 不将旧清单按 B1 绑定的成绩直接改名为此权重的成绩；本仓库[阶段图表](../../docs/stage-results.md)按各自评测来源记录。
- 公开包中适配器配置的旧机器路径已替换为官方模型 ID。下载后用随包的 `SHA256SUMS` 校验。

## 本地训练基座

训练入口也可将完整底座放置为：

```text
models/
└── Qwen3-4B-Instruct-2507/
    ├── config.json
    ├── model.safetensors.index.json
    ├── model-*.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

同时保留快照附带的其他配置与模型许可证。不要改写基座 tokenizer 或聊天模板；训练与评测使用相同快照。各入口也可通过 `--model-path` 指定另一处完整本地路径。

程序按本地文件加载，不自动下载模型。训练快照没有可核实的 resolved revision，因此使用文件 SHA-256 固定版本，不填造版本号。随底座保留原始模型卡及 Apache-2.0 许可证。

→ [适配器与后续权重发布](../rp_grpo/checkpoints/README.md)
