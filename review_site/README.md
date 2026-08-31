# MitoEvidence 标注审阅台

这是四份冻结 JSONL 的只读展示层，不是新的标注来源。构建前会逐文件核对
`expert_gold_manifest.json` 中的 SHA-256、记录数和唯一 ID。

```bash
python scripts/build_annotation_review_site.py
python scripts/build_annotation_review_site.py --check
python -m http.server 8765 --directory review_site
```

页面使用相对资源路径，可部署在 GitHub Pages 的 `/hy-agent/` 子路径。所有来自源记录的文本均以
`textContent` 写入 DOM，不会作为 HTML 执行。
