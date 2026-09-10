# 本地基座

基座为 `Qwen/Qwen3-4B-Instruct-2507`。权重不随 Git 仓库分发；从模型提供方获取完整快照后，放置为：

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

程序按本地文件加载，不自动下载模型。发布训练结果时记录模型 ID、实际 revision（如可获得）、文件 SHA-256、tokenizer 与训练配置；不填造不可核实的版本值。

→ [适配器与后续权重发布](../rp_grpo/checkpoints/README.md)
