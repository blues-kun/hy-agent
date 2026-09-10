# 适配器与权重发布

最终强化学习权重由项目后续补充。本仓库当前不包含基座、旧工具适配器或新 RL 权重，也没有预设下载地址。

训练默认写入 `mito/runs/`：

| 产物 | 运行目录中的位置 |
| --- | --- |
| 核验 SFT | `adapter/` |
| 工具 SFT | `adapter/` |
| GRPO / RP-GRPO | `adapter/policy/` |

如需从已有工具 SFT 继续，显式传入 `--initial-adapter`；不传时工具 SFT 从本地基座初始化 LoRA。正式 RL 需使用通过当前工具协议准入的工具 SFT 起点。

后续可按以下结构整理发布材料，目录在真实权重就绪后创建：

```text
checkpoints/<release-id>/
├── adapter_config.json
├── adapter_model.safetensors
├── MODEL_CARD.md
└── release_manifest.json
```

`release_manifest.json` 建议记录：训练角色与组别、基座 ID / revision / 哈希、初始 SFT 哈希、代码 commit、数据清单哈希、训练配置、随机种子、工具协议与准入报告哈希、评测记录和每个发布文件的 SHA-256。核验与工具策略分别发布，不共用含混的模型名称。

权重文件可使用正式 Release 或模型仓库分发；确定实际地址后再补链接。checkpoint 的权重续接不等同于完整优化器与随机状态恢复。
