# 训练图像

| 文件 | 来源 | 内容 |
| :--- | :--- | :--- |
| [mito-training-route.svg](mito-training-route.svg) | 项目训练路线与本次补充的专家反馈流程 | 知识库适配、双分支 SFT、RP-GRPO 与 Hy3 接入 |
| [sft-loss.png](sft-loss.png) | 项目提供的 `loss.png` | Qwen3-4B SFT 训练损失与验证集示范损失 |
| [first200-success-comparison.png](first200-success-comparison.png) | 项目提供的前 200 步 Excel 展示页，使用 R 绘制 | 20 题监测集的 avg@4 与双侧配对成功率 |
| [full-validation-step200.png](full-validation-step200.png) | 同批完整验证集与 SFT 对照记录 | 74 题完整验证集的四组 200 步结果、SFT100 与 SFT400 |
| [training-dynamics.png](training-dynamics.png) | 同批训练日志，使用 R 绘制 | 此前 25 步窗口均值的训练奖励与策略损失 |

三张新图均提供同名 SVG 与 PDF。PNG 为 600 dpi；方法使用固定配色与点形，无额外平滑、缺失补点或跨种子置信带。SFT100 为共同起点，SFT400 为独立对照，不是 RL 第 400 步。

监测图采用项目指定的 Excel 展示标签；展示页与原始字段的 B1/B3 归属冲突仍保留在[阶段记录](../../docs/stage-results.md#数据来源与已知问题)中，未通过改名或改分处理。

`sft-loss.png` 保持原图，未裁剪或重绘，不表示 RP-GRPO 损失；旧阶段汇总在阶段记录中保留，不与本轮精确步数拼接。
