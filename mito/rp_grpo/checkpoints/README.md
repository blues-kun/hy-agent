# 适配器与权重发布

本仓库不包含大权重。原始 Qwen 底座与项目确认的 **RP-GRPO step 200 最终 policy LoRA** 已整理为两个独立推理发布包，下载入口与加载示例见[模型权重](../../models/README.md)。网盘真实地址确定后再补链接，不放占位下载 URL。

公开包只保留底座、最终适配器及必要配置、许可证、模型卡和校验信息；`reference/`、`training_state.pt`、优化器／随机状态和内部数据不进入推理包。最终 policy 已包含 SFT 初始化后继续 RL 训练的参数，推理不再叠加旧 SFT。

源检查点旧标签的更正依据来自项目负责人，在发布清单中单独记录，原始张量与源文件保留。200 为采样循环与更新数，不是 epoch；文件结构和有限值检查不等于完整推理或工具评测通过。

训练默认写入 `mito/runs/`：

| 产物 | 运行目录中的位置 |
| --- | --- |
| 核验 SFT | `adapter/` |
| 工具 SFT | `adapter/` |
| GRPO / RP-GRPO | `adapter/policy/` |

如需从已有工具 SFT 继续，显式传入 `--initial-adapter`；不传时工具 SFT 从本地基座初始化 LoRA。正式 RL 需使用通过当前工具协议准入的工具 SFT 起点。

独立权重存储中使用以下结构，Git 中仅维护本文档：

```text
checkpoints/<release-id>/
├── adapter_config.json
├── adapter_model.safetensors
├── MODEL_CARD.md
└── release_manifest.json
```

`release_manifest.json` 建议记录：训练角色与组别、基座 ID / revision / 哈希、初始 SFT 哈希、代码 commit、数据清单哈希、训练配置、随机种子、工具协议与准入报告哈希、评测记录和每个发布文件的 SHA-256。核验与工具策略分别发布，不共用含混的模型名称。

权重文件可使用 Google Drive 等网盘或模型仓库分发；确定实际地址后再补链接。checkpoint 的权重续接不等同于完整优化器与随机状态恢复。
