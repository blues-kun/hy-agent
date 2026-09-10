# MITO Agent 仓库迁移 · 2026-09-10

## 内容调整

项目首页由早期 MitoEvidence 证据评测方案更新为 MITO Agent 项目展示：线粒体表型解析、数字人实验协同、文献知识与机制探索、实验辅助规划。

当前源码以 `mito/` 为统一训练目录，包含独立证据核验 SFT、工具策略 SFT、GRPO / RP-GRPO、真实只读工具环境、准入与 B0–B3 / LOO 比较入口。该目录替换旧评测实现，不替换线上 Web/API、影像执行服务或完整知识库。

## 发布范围

| 随仓库提供 | 单独维护 |
| :--- | :--- |
| 训练、比较、工具接口及测试源码 | Qwen 基座权重、旧工具适配器 |
| 核验标注 train 443 / dev 51、输出契约、文件校验清单 | 最终强化学习权重及对应实验结果，后续补充 |
| 17 PNG + 9 GIF 原始展示素材，含数字人摄像头和大型 GIF | 内部全文、实验测量、轨迹、审计原始记录 |
| 历史 127 条标注及浏览页面 | 本地模型缓存、运行产物与凭据 |

新标注公开导出只将 `metadata.original_record.file` 缩为源文件名，移除内部绝对目录；不更改 prompt、completion、labels、分区和原始内容哈希。`mito/data/MANIFEST.json` 同时记录源文件哈希与公开文件的新哈希。导出脚本为 [`export_public_annotations.py`](../mito/code/export_public_annotations.py)。

## Git 与运行约定

- 旧版提交 `24140835df50cfb1837dd188e4d40433e5f9ddbc` 已保存到 [legacy/mitoevidence-20260831](https://github.com/blues-kun/hy-agent/tree/legacy/mitoevidence-20260831)，历史代码与结果仍可查询。
- 主分支采用常规增量提交，不重写历史，不使用强制推送。
- 文本采用 UTF-8 / LF；标注及 JSON 清单按原始字节校验；图片与 GIF 作为二进制追踪。
- 权重、内部快照、运行目录及凭据已加入忽略规则。最终权重入口见 [`mito/models/README.md`](../mito/models/README.md)。
- CPU 测试与依赖私有快照或 GPU 的检查分开；本次迁移不执行 SFT / RL 训练。

## 后续权重接入

补充最终适配器时，一并提供基座版本、训练配置、代码提交、数据分区与文件哈希。权重存储于模型仓库或发布附件，Git 中记录获取位置与校验信息；实际工具执行与独立开发/测试结果按版本关联。

## 本次验证

Windows / Python 3.12.14 独立环境下，公共源码测试 **211 passed / 76 skipped**。跳过项分别需要私有 RP 快照（49）、本地 tokenizer（2）、可选 torch（3）、POSIX（20）或 Windows symlink 权限（2）；不将跳过计入通过。

另以原始 RP 快照进行只读集成验证，**66 passed**，不复制该快照。7 个 CLI 帮助入口、标注浏览页构建校验及 JavaScript 语法检查通过；26 份展示媒体与源文件 SHA-256 一致，README 本地图片与文档链接检查通过。上述均为工程验证，不是模型效果或新的训练结果。
