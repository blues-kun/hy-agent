<div align="center">

# MITO Agent

### 面向线粒体研究的干湿闭环科研智能体

从显微图像到表型量化，从文献证据到机制假设与实验规划。

[![Application](https://img.shields.io/badge/MITO_Agent-在线体验-171717?style=flat-square)](https://agent.blueskun.com:8444/)
[![Training](https://img.shields.io/badge/Training-SFT_·_RP--GRPO-404040?style=flat-square)](mito/)
[![Annotations](https://img.shields.io/badge/Annotations-Train_443_·_Dev_51-666666?style=flat-square)](mito/data/)
[![Weights](https://img.shields.io/badge/RL_Weights-待补充-888888?style=flat-square)](mito/models/README.md)

<a href="#项目概览">项目概览</a> · <a href="#功能展示">功能展示</a> · <a href="#模型与训练">模型与训练</a> · <a href="#代码与数据">代码与数据</a> · <a href="assets/showcase/README.md">完整展示</a>

</div>

---

## 项目概览

MITO Agent 聚焦胰岛与 β 细胞线粒体研究，将显微成像分析、文献知识和实验设计组织到同一个工作流中。研究者可通过文字、图像或语音提出任务，查看量化结果与来源证据，修订实验方案，再将实验反馈带回下一轮分析。

| 线粒体表型解析 | 数字人实验协同 | 知识增强与实验规划 |
| :--- | :--- | :--- |
| 图像降噪、实例分割、形态与功能量化、分析报告 | 语音与视觉交互、说话人识别、实验过程辅助 | 文献检索、图谱关系、证据核验、候选机制与方案草拟 |

<p align="center">
  <img src="assets/mito-workflow.svg" width="100%" alt="MITO Agent 研究流程：采样成像、表型解析、证据与机制、实验规划、湿实验反馈" />
</p>

### 当前研究积累

| 胰岛样本 | 线粒体分析 | 知识关系 | 候选关系 |
| :---: | :---: | :---: | :---: |
| **378** | **约 12.7 万** | **2,234** | **732** |

以上为项目展示材料记录的应用规模；候选关系用于进一步筛选和实验验证，不是已证实的新机制，也不代表强化学习成绩。

## 功能展示

### 01 · 线粒体表型解析

面向自研成像体系，采用二维结构盲点自监督降噪与融合形态先验的弱监督分割，连接形态、动态及适用的功能指标分析，并生成可检查的图表与报告。

![线粒体分割比较](assets/showcase/segmentation-comparison.png)

<p align="center"><sub>从左到右：原图、MoDL、Nellie、Omnipose、自研算法。下排为对应局部放大。</sub></p>

<details>
<summary>查看图像解析工作流与量化报告</summary>

![图像检查与分割工作流](assets/showcase/image-analysis.gif)

![线粒体功能量化报告](assets/showcase/quantitative-report.png)

图像工作流展示样本检查、处理建议与分割候选结果；报告示例展示寿命分布与统计汇总。

</details>

### 02 · 文献知识与机制探索

将文献、实体关系和原文证据放在同一研究上下文中。图谱提供关联线索，大模型组织研究问题与解释，小模型承担领域核验；文献事实、模型候选与实验观察分别记录。

<table>
  <tr>
    <td width="50%"><img src="assets/showcase/knowledge-graph.png" alt="知识图谱与证据定位" /></td>
    <td width="50%"><img src="assets/showcase/model-collaboration.png" alt="大小模型协同分析" /></td>
  </tr>
  <tr>
    <td align="center"><sub>知识图谱与证据定位</sub></td>
    <td align="center"><sub>大小模型协同分析</sub></td>
  </tr>
</table>

<details>
<summary>查看文献检索与候选机制</summary>

![文献检索与辅助阅读](assets/showcase/literature-search.gif)

![候选机制与来源证据](assets/showcase/mechanism-hypotheses.png)

</details>

### 03 · 实验规划与协同

围绕研究假设生成基于证据与规则的实验草案，组织对照、独立重复和机制验证。数字人提供语音与视觉交互，研究者确认方案并开展实验，将观察结果用于修订候选解释。

![基于证据与规则的实验规划](assets/showcase/experiment-planning.png)

<details>
<summary>查看数字人协同实验与湿实验成像</summary>

![数字人协同实验](assets/showcase/digital-human.gif)

![OPP、线粒体与融合成像](assets/showcase/wet-lab-imaging.png)

</details>

全部 **26 份原始图片与 GIF** 见[完整展示](assets/showcase/README.md)，包括成像采集、交互分析、降噪比较与实验协同。

## 模型与训练

大模型负责理解任务与综合结果；工具规划模型决定下一步动作；核验模型检查主张、实验条件和证据支持。知识库保留直接检索到大模型的路径，工具由服务端执行并返回可追溯结果。核验与规划分别训练，一个任务只保留一个动作决策者。

| 模块 | 学习目标 | 实现入口 |
| :--- | :--- | :--- |
| **Verifier SFT** | 根据给定证据核验主张，输出支持状态、条件缺口与必要改写 | [`train_verifier.py`](mito/code/train_verifier.py) |
| **Tool SFT** | 选择工具、填写参数、读取实际返回并继续或停止 | [`train_planner_sft.py`](mito/rp_grpo/train_planner_sft.py) |
| **GRPO / RP-GRPO** | 通过实际工具执行优化成功率、条件适应与调用成本 | [`train_policy.py`](mito/rp_grpo/train_policy.py) |

**RP-GRPO** 对同一科研任务构造不同的可观察证据条件，分别执行工具轨迹，以平均收益和较弱侧收益共同优化行动决策。

| 对照组 | 配置 | 关注问题 |
| :--- | :--- | :--- |
| B0 | 工具 SFT + 固定科学规则 | 规则策略是否已足够 |
| B1 | 工具 SFT + GRPO | 基础强化学习的作用 |
| B2 | GRPO + 配对任务日程 | 配对数据组织的作用 |
| B3 | RP-GRPO + 同一配对日程 | 配对优化目标的增量 |
| LOO | η = 0，固定尺度 leave-one-out | 区分配对目标与优势估计改动 |

训练代码、工具接口和比较入口已提供。**最终强化学习权重、对应训练配置与评测结果后续补充。**

方法与运行说明：[训练指南](mito/README.md) · [RP-GRPO 方法](mito/rp_grpo/RP_METHOD.md) · [工具接口](mito/rp_grpo/INTERFACES.md) · [模型与权重](mito/models/README.md)

## 代码与数据

```text
hy-agent/
├── mito/                    # 核验 SFT、工具 SFT、GRPO / RP-GRPO
│   ├── code/                # 核验训练、对比与数据导出
│   ├── rp_grpo/             # 工具环境、策略训练与准入检查
│   ├── data/                # 已标注 train / dev 与格式契约
│   ├── models/              # 基座及后续权重接入说明
│   └── tests/               # 算法、数据与流程测试
├── assets/showcase/         # 项目原始图片与 GIF
├── annotation_prelabel/     # 历史 127 条专家参考标注
└── review_site/             # 历史标注浏览页面
```

本次发布包含训练实现、标注数据与项目展示；线上 Web/API、影像服务及完整知识库独立维护。基座权重、旧适配器、内部原文快照与运行日志不进入 Git。

```bash
git clone https://github.com/blues-kun/hy-agent.git
cd hy-agent/mito
python -m pip install -r requirements-test.txt
python -m pytest -q
```

上述入口执行公共 CPU 测试，不启动训练。GPU 训练及完整快照测试见[训练指南](mito/README.md)。

标注数据：[核验训练集与开发集](mito/data/README.md) · [历史标注说明](annotation_prelabel/README.md) · [在线标注浏览](https://blues-kun.github.io/hy-agent/)

版本记录：[本次迁移](docs/migration-20260910.md) · [上一版 MitoEvidence](https://github.com/blues-kun/hy-agent/tree/legacy/mitoevidence-20260831)

---

<p align="center"><sub>MITO Agent · Imaging · Evidence · Experiment</sub></p>
