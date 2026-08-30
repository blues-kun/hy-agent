# 三线核验报告：文献引用 · 金标语料 · 方法学

> 日期：2026-08-28
> 方法：三个并行研究代理分别核验（文献核验 / 金标语料与数据源 / 方法学），主代理对两个关键项亲自复核（npj Digital Medicine 论文原文、AgentGL 的 ACL Anthology 页面均直接打开确认）。
> 铁律：每条结论附实际打开的 URL；未核实项明确标注。

> **执行口径更新（2026-08-31）：** 本文保留的是立项时的方法学核验记录。项目负责人已确认
> `annotation_prelabel/` 中的 127 条记录为本轮唯一的单一专家共识金标，不再等待第二位专家；
> 因此下文“双人评分、第三人裁决、双人金标”等内容只用于未来扩展设计，不是当前实验的
> 完成条件。当前只报告 system-vs-expert-consensus，不报告 inter-rater κ/ICC。

---

## 0. 对方案的直接决定（TL;DR）

| # | 结论 | 状态 |
|---|---|---|
| 1 | **评估器架构定案**：单推理 Judge + 校准 + 跨家族审计，不做多 Agent 共识评分。直接实证：Croxford 等（npj Digital Medicine 2025）单 o3-mini Judge ICC 0.818 > 最佳多 Agent 0.768，且多 Agent 共识"过早收敛于偏高分数" | 已写入方案 7.3 |
| 2 | **未来双人评分扩展的样本量参考**：κ=0.70 vs 0.5 需 n=69（80% 功效）/ n=89（90% 功效） | 已写入方案 10.5；本轮不执行双人评分 |
| 3 | **金标语料就绪**：12 篇 β 细胞线粒体综述（2015–2026，11 种期刊）+ 4 篇备选，合计约 2,290 条参考文献可作 gold 证据池；其中 8/12 可走 Europe PMC fullTextXML 全文通道 | 语料路线可开工 |
| 4 | **引用核验流水线参数确定**：Crossref 批量 filter 50 DOI/请求（约 150 DOI/s，比逐条快 15 倍）；NCBI 申请 API key（10 rps）+ tool/email 邮件登记；Europe PMC 无官方限流数字，保守 1–2 并发；Annotations API **无字符 offset**，定位走文本锚点 | 流水线设计依据 |
| 5 | **GRPO 不做的实据**：官方配方 GRPO 实测 128×H20（16 节点），rollout 单实例最低 TP=16；全参 SFT 32×80GB；LoRA 也需单机 8×80GB | 已写入方案 5.4 |
| 6 | **引用修正三处**：Medical Graph RAG 须用 ACL 正式版标题（"Evidence-based…"，非 arXiv 版"Towards Safe…"）；KARL 正式标题不含"Knowledge-Augmented Reinforcement Learning"；AgentGL 实际存在（此前"检索不到"的结论作废，系 ACL 2026 论文集 7 月才上线） | 写方案/报告时照此引用 |
| 7 | **无撞车**：与本方案最接近的是 MedRAGChecker（生物医学 RAG 的 claim 级核验），但它面向短答 QA、不做实验条件维度；"综述场景 + 物种/细胞/剂量/效应方向条件保真评估器"仍是空白。相关工作须引用 MedRAGChecker、CliniFact 并写明差异 | 分析报告相关工作章节素材 |

---

## 1. 文献核验结果（Prompt 2）

### 1.1 逐条核验表

| 引用 | 状态 | 规范引用 | 链接 |
|---|---|---|---|
| RAGChecker | 已核实 | RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation. Dongyu Ru 等（Amazon+SJTU+西湖）. NeurIPS 2024 Datasets & Benchmarks | [arXiv:2408.08067](https://arxiv.org/abs/2408.08067)、[NeurIPS poster](https://neurips.cc/virtual/2024/poster/97761)、[GitHub](https://github.com/amazon-science/ragchecker) |
| 临床 Judge 论文 | 已核实（主代理复核原文） | Evaluating clinical AI summaries with large language models as judges. Emma Croxford 等（UW-Madison）. npj Digital Medicine 8, 640 (2025) | [DOI 10.1038/s41746-025-02005-2](https://doi.org/10.1038/s41746-025-02005-2) |
| Medical Graph RAG | 已核实，注意版本标题差异 | Medical Graph RAG: **Evidence-based** Medical Large Language Model via Graph Retrieval-Augmented Generation. Junde Wu（牛津）. ACL 2025, pp. 28443–28467（arXiv 版副标题为 "Towards Safe…"，引用以 ACL 版为准） | [2025.acl-long.1381](https://aclanthology.org/2025.acl-long.1381/)、[arXiv:2408.04187](https://arxiv.org/abs/2408.04187) |
| PaperQA2 / LitQA2 | 已核实 | Language agents achieve superhuman synthesis of scientific knowledge. Michael D. Skarlinski 等（FutureHouse）. arXiv 预印本（未见正式版），LitQA2 为其 248 题基准 | [arXiv:2409.13740](https://arxiv.org/abs/2409.13740) |
| ASReview | 已核实 | An open source machine learning framework for efficient and transparent systematic reviews. Rens van de Schoot 等. Nature Machine Intelligence 3, 125–133 (2021) | [DOI 10.1038/s42256-020-00287-7](https://www.nature.com/articles/s42256-020-00287-7) |
| Biolink Model | 已核实 | Biolink Model: A universal schema for knowledge graphs in clinical, biomedical, and translational science. Deepak R. Unni 等. Clin Transl Sci 15(8):1848–1855 (2022) | [DOI 10.1111/cts.13302](https://doi.org/10.1111/cts.13302) |
| PrimeKG | 已核实 | Building a knowledge graph to enable precision medicine. Payal Chandak 等. Scientific Data 10, 67 (2023) | [DOI 10.1038/s41597-023-01960-3](https://www.nature.com/articles/s41597-023-01960-3) |
| AgentGL | 已核实（主代理复核页面） | AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning. Yuanfu Sun 等. ACL 2026, pp. 25313–25335, DOI 10.18653/v1/2026.acl-long.1161 | [2026.acl-long.1161](https://aclanthology.org/2026.acl-long.1161/) |
| KARL | 已核实，标题须修正 | KARL: Reinforcement Learning for LLM Agents on Multi-Turn Knowledge-Intensive Agentic Tasks. Xueqiao Sun 等. ACL 2026, pp. 47539–47558（"Knowledge-Augmented RL"只是摘要里的缩写展开，不在标题中） | [2026.acl-long.2196](https://aclanthology.org/2026.acl-long.2196/) |
| ToolOmni | 已核实 | ToolOmni: Enabling Open-World Tool Use via Agentic learning with Proactive Retrieval and Grounded Execution. Shouzheng Huang 等. ACL 2026, pp. 37421–37439（"learning"小写照抄） | [2026.acl-long.1736](https://aclanthology.org/2026.acl-long.1736/) |

### 1.2 架构决定项：单 Judge vs 多 Agent（详细证据）

**Croxford 等，npj Digital Medicine 2025**（主代理已打开原文逐段核对）：

- 金标：经心理测量学验证的 PDSQI-9 量表（9 属性），7 名不同专科医生对 200 份真实 EHR 多文档摘要评分（量表本身 ICC 0.867）；
- 对比：单 Judge（GPT-4o / GPT-o3-mini / DeepSeek R1 / Mixtral 8×22B / Llama 3.1 8B，zero-shot/few-shot/SFT/DPO）vs 多 Agent（AutoGen MagenticOneGroupChat：1 orchestrator + 3 评审 agent，医生人格或高/中/低立场两种 persona 方案）；
- 主指标 ICC(3,k)；比较方式为 7 名人类中位数 vs 同一 Judge 独立 7 次的中位数；
- 结果：最佳单 Judge = GPT-o3-mini（5-shot）**ICC 0.818**（95% CI 0.772–0.854），单次 22 秒、约 5 美分；最佳多 Agent（全 o3-mini）**ICC 0.768**（95% CI 0.710–0.814）；
- 摘要原文："reasoning models excelled in inter-rater reliability … outperforming non-reasoning, task-trained, and multi-agent approaches"；
- 机制证据（原文案例）：即使三个 agent 全部是 o3-mini，多 Agent 共识也会"过早收敛到偏高分数"（人类中位 3 分的 Organized 属性，多 Agent 给 5 分），单 Judge 反而对齐人类中位数；
- 附带发现：GPT-4o 当 Judge 评自家生成反而更苛刻，来源模型不影响 ICC（p>0.2）——self-preference 在该实验中未成偏差主因，但仍需按审计规范报告。

**配套校准方法**：Rouillard 等（[arXiv:2604.14892](https://arxiv.org/abs/2604.14892)，2026-04）——LLM Jury 对 3,334 个真实诊断打分，未校准分数保序但系统性偏低；在 330 例专家标注上做**等张回归（isotonic regression）事后校准**后与专家小组一致性达到甚至超过人类复评小组。操作含义：用我们 10 题校准集 + 双人金标做 Judge 事后校准即可，无需多 Agent。

**结论**：方案 7.3 的"单一冻结配置 Hy3 Judge + 规则 + 人工复核"路线获得直接实证支持；跨家族第二 Judge（Qwen/DeepSeek）只做偏差审计，不做共识评分。

### 1.3 2025H2–2026 新工作扫描（撞车与补引分析）

**Claim 级核验（生物医学）**

1. **MedRAGChecker: Claim-Level Verification for Biomedical Retrieval-Augmented Generation**（Yuelyu Ji，匹兹堡，[arXiv:2601.06519](https://arxiv.org/abs/2601.06519)，2026-01）——原子 claim 分解 + NLI + 知识图谱一致性，区分检索/生成失败。**最接近的相关工作，必须引用**；差异：它面向短答 QA，不做实验条件（物种/细胞/剂量/效应方向）维度，不面向综述。
2. **CliniFact**（Scientific Data 12, 86 (2025)，[DOI 10.1038/s41597-025-04417-x](https://www.nature.com/articles/s41597-025-04417-x)）——从 ClinicalTrials.gov 构造 1,970 条含干预/对照/结局/统计字段的临床 claim。其"claim = 结构化条件字段"思路与本方案 D4 同源，构造方法可借鉴。
3. FActBench（[arXiv:2509.02198](https://arxiv.org/abs/2509.02198)）——医学生成任务事实核查基准；CoT+NLI 一致投票与专家相关性最高。
4. VerifAI（[arXiv:2604.08549](https://arxiv.org/abs/2604.08549)）——开源生物医学 QA 的原子 claim + 微调 NLI 核验。

**引用忠实性 / 归因**

5. **CiteGuard**（ACL 2026，[2026.acl-long.282](https://aclanthology.org/2026.acl-long.282/)）——检索增强的引用归因验证；实证指出**裸 LLM Judge 做引用判断 recall 仅 16–17%**，直接支持本方案"引用核验必须接检索与规则、不能只靠 Judge"的设计。
6. CiteEval（ACL 2025，[2025.acl-long.1574](https://aclanthology.org/2025.acl-long.1574/)）——论证纯 NLI 支持度是引用评估的次优代理，提出全上下文细粒度框架。
7. VeriCite（[arXiv:2510.11394](https://arxiv.org/abs/2510.11394)）——三阶段 RAG 引用核验与精修。

**LLM Judge 校准**

8. Rouillard 等（[arXiv:2604.14892](https://arxiv.org/abs/2604.14892)）——见 1.2，等张回归事后校准。
9. Croxford 等（npj Digital Medicine 2025）——见 1.2。
10. LLM-as-a-Judge in Healthcare 范围综述（[arXiv:2605.25273](https://doi.org/10.48550/arxiv.2605.25273)，134 项研究）——相关工作定位用。

**撞车结论**：专门针对"医学综述场景 + 实验条件级保真"的 claim 级评估器未检索到（最接近的只有毒理学信息抽取管线，是抽取器不是评估器）。**方案的差异化空间成立**。

---

## 2. 金标语料（Prompt 3.1）

### 2.1 主池：12 篇已核验综述（约 2,043 条参考文献）

核验方式：PubMed 检索页发现 → NCBI esummary 核对元数据 → Europe PMC REST 核对 PMCID/OA；参考文献数来自 `…/MED/<pmid>/references` 实测。正式入池前应跑本项目 L1 核验脚本再冻结（吃自己的狗粮）。

| # | 标题（缩写） | 第一作者 | 年 | 期刊 | PMID | OA/PMCID | 参考文献数 | 覆盖 |
|---|---|---|---|---|---|---|---|---|
| 1 | Mitochondrial regulation of β-cell function… | Kaufman BA | 2015 | Mol Aspects Med | [25659350](https://pubmed.ncbi.nlm.nih.gov/25659350/) | ✅ PMC4404204 | 144 | mtDNA、生物合成、代谢偶联、动力学总览 |
| 2 | Mitochondrial network regulation… inflammatory signals | Baltrusch S | 2016 | Diabetologia | [26873508](https://pubmed.ncbi.nlm.nih.gov/26873508/) | ❌ | 53 | fission-fusion × 炎症（IL-1β） |
| 3 | Transcribing β-cell mitochondria in health and disease | Mulder H | 2017 | Mol Metab | [28951827](https://pubmed.ncbi.nlm.nih.gov/28951827/) | ✅ PMC5605719 | 262 | mtDNA 转录（TFAM/Tfb1m）、OxPhos、T2D |
| 4 | Metabolic and functional specialisations of the beta cell | Rutter GA | 2020 | Diabetologia | [32894309](https://pubmed.ncbi.nlm.nih.gov/32894309/) | ✅ PMC7476987 | 62 | disallowed genes、代谢偶联、胰岛 Ca²⁺ 同步 |
| 5 | Mitochondrial Calcium Signaling in Pancreatic β-Cell | Weiser A | 2021 | Int J Mol Sci | [33802289](https://pubmed.ncbi.nlm.nih.gov/33802289/) | ✅ PMC7959128 | 119 | MCU/NCLX、ER-线粒体耦合、GSIS |
| 6 | A Selective Look at Autophagy in Pancreatic β-Cells | Pearson GL | 2021 | Diabetes | [34016598](https://pubmed.ncbi.nlm.nih.gov/34016598/) | ⚠️ PMC8275885 非 OA（fullTextXML 实测 404） | 130 | 选择性自噬/mitophagy（CLEC16A 等） |
| 7 | Mitochondrial Dynamics and Insulin Secretion | Kabra UD | 2023 | Int J Mol Sci | [37762083](https://pubmed.ncbi.nlm.nih.gov/37762083/) | ✅ PMC10530730（fullTextXML 实测 200，229KB） | 136 | Drp1/Mfn1-2/Opa1 核心机器 |
| 8 | Mitochondrial bioenergetics, metabolism, and beyond… | Rivera Nieves AM | 2024 | Front Mol Biosci | [38404962](https://pubmed.ncbi.nlm.nih.gov/38404962/) | ✅ PMC10884328 | 248 | TCA/OxPhos、膜电位、ROS、Ca²⁺ 全景 |
| 9 | Physiological Fatty Acid-Stimulated Insulin Secretion… | Ježek P | 2025 | Antioxid Redox Signal | [39834189](https://pubmed.ncbi.nlm.nih.gov/39834189/) | ❌ | 489 | 脂肪酸分泌、redox、糖脂毒性（引文最多但无 PMC 全文，引文列表走 Crossref reference 字段） |
| 10 | …ER-Mitochondria Redox Balance | Zaher A | 2025 | Cells | [40136648](https://pubmed.ncbi.nlm.nih.gov/40136648/) | ✅ PMC11941261 | 107 | MAM/Ca²⁺/氧化还原/UPR |
| 11 | Mitophagy in the adaptation to pancreatic β cell stress | Levi-D'Ancona E | 2026 | Trends Endocrinol Metab | [41109799](https://pubmed.ncbi.nlm.nih.gov/41109799/) | ❌ | 123 | PINK1/Parkin、CLEC16A、T1D+T2D |
| 12 | Mitochondrial Homeostasis in Pancreatic β Cell Function | Li R | 2026 | J Diabetes | [42051080](https://pubmed.ncbi.nlm.nih.gov/42051080/) | ✅ PMC13125871 | 170 | 质量控制、细胞器互作、糖脂毒性 |

备选池：Darwish R 2025（[41369350](https://pubmed.ncbi.nlm.nih.gov/41369350/)，PMC12691418，247 refs）、Muñoz F 2024（[38656044](https://pubmed.ncbi.nlm.nih.gov/38656044/)，159 refs）、An Y 2026（[41827908](https://pubmed.ncbi.nlm.nih.gov/41827908/)，PMC12984212，141 refs）、Kavyashree S 2025（[40204078](https://pubmed.ncbi.nlm.nih.gov/40204078/)，115 refs）。

**争议题特供**：Diabetes 2024 同期 point-counterpoint——Rutter & Sweet（[38768365](https://pubmed.ncbi.nlm.nih.gov/38768365/)，PMC11109788）vs Merrins & Kibbey（[38768366](https://pubmed.ncbi.nlm.nih.gov/38768366/)，PMC11109790），就 K_ATP/OxPhos 经典模型是否需重构正面交锋。引文少不适合当证据池，但**是"冲突与异质性分析"题型和"学界未定论断言"难例的绝佳素材**。

### 2.2 构造提醒

- 12 篇中只有 **8 篇**可走 Europe PMC fullTextXML（判据必须用 `isOpenAccess=Y`，有 PMCID 不等于可取全文，实测非 OA 的 PMC8275885 返回 404）；
- 无 PMC 全文的引文列表改走 Crossref `reference` 字段或出版商侧；
- 机制覆盖互补性已核查：动力学（#2/#7）、mitophagy（#6/#11）、Ca²⁺（#4/#5/#10）、代谢偶联（#3/#4/#8）、redox/膜电位（#8/#9/#10）、糖脂毒性（#9/#12）、mtDNA 转录（#1/#3）。

---

## 3. 数据源 API 事实（Prompt 3.2/3.3）

### 3.1 Europe PMC

| 项 | 事实 | 来源 |
|---|---|---|
| 限流数字 | **官方无任何数字**（四个官方页面 + 参考指南 v1.51 全文检索均无；响应头无 x-rate-limit）。流传的"10 req/s"全部出自第三方，不得写入设计文档为事实 | [RestfulWebService](https://europepmc.org/RestfulWebService)、[developers](https://europepmc.org/developers)、[参考指南 PDF](https://europepmc.org/docs/EBI_Europe_PMC_Web_Service_Reference.pdf) |
| 唯一硬条款 | 禁爬虫批量抓取网站；自动化只允许 OAI/RESTful/SOAP/bulk download 四类通道 | [copyright](https://europepmc.org/copyright) |
| 全文获取 | `…/rest/{PMCID}/fullTextXML` 仅 OA 子集（实测 OA 200 / 非 OA 404）；OA 语料按 PMCID 区间打包于 [FTP](https://europepmc.org/ftp/oa/) | [downloads](https://europepmc.org/downloads) |
| **PMID-PMCID-DOI 映射文件** | 每月 1 日覆盖更新，**ID 互查全部离线化的关键**，评测流水线必用 | 同上 |
| Annotations API | **无字符 offset**！定位=W3C TextQuoteSelector（prefix/exact/postfix 三段文本锚点）+ section 受控词表（18 种 section、44 种注释类型）；`section=Results&type=Gene_Proteins` 可区分"结果段原创断言"与"引言背景转述" | [AnnotationsApi](https://europepmc.org/AnnotationsApi) |
| 复现机制 | 响应带 API `version` 与完整 `request` 回显、`cursorMark` 深分页、pageSize 1–1000、POST 端点 `/searchPOST`；官方无专门快照指南，自组：落盘 {version, request 回显, 日期戳, hitCount} + 冻结 PMID 列表 + FTP 快照 | 实测 + 参考指南 |

### 3.2 NCBI E-utilities

- 限流：无 key **3 rps**，有 key **10 rps**（更高需邮件申请）；`tool`+`email` 参数必填且**必须另发邮件到 eutilities@ncbi.nlm.nih.gov 登记**，否则仍可能封 IP；
- 批量：>200 UID 用 POST；epost 上限 10,000 UID；esummary/efetch `retmax` 上限 10,000；history server（WebEnv+query_key）流程实测可用；
- 大批量作业建议周末或美东 21:00–05:00；
- 文档：[NBK25497](https://www.ncbi.nlm.nih.gov/books/NBK25497/)、[NBK25499](https://www.ncbi.nlm.nih.gov/books/NBK25499/)。

### 3.3 Crossref

- **文档滞后警告**：官方表格写 polite 10 rps，但 2026-08-28 实测已按请求类型分池——`polite-single` 10 rps / `polite-array`（filter 列表查询）**仅 3 rps**；必须按响应头 `x-rate-limit-limit`/`x-concurrency-limit` 动态自适应，429 退避，403=人工封禁；
- polite pool：User-Agent 带 `mailto:` 或 query 参数 `?mailto=`，实测均生效；
- **批量核验最优解**：`/works?filter=doi:A,doi:B,…&select=DOI,title,container-title,issued&rows=100`，实测单请求 50 DOI 全中（URL 1,760B）；折算吞吐 150 DOI/s，**比逐条 /works/{doi} 快约 15 倍**；500 条引用约 10 请求 4 秒；
- 分页：`rows` 最大 1000；深分页 `cursor=*`；增量同步用 `from-index-date` 且必须配 `until-index-date`；
- 文档：[access & authentication](https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/)、[tips](https://www.crossref.org/documentation/retrieve-metadata/rest-api/tips-for-using-the-crossref-rest-api/)。

### 3.4 流水线三条落地原则

1. ID 互查离线化（Europe PMC 月度映射文件），元数据核验 Crossref 批量 filter + NCBI esummary POST，500 条规模秒级完成；
2. 证据定位基于文本锚点（prefix/exact/postfix 在全文 XML 中重定位），不基于字符偏移；EvidenceSpan Schema 据此设计；
3. 限流按响应头自适应，不写死常量；Europe PMC 保守 1–2 并发。

---

## 4. 方法学（Prompt 4）

### 4.1 κ 功效与 Gwet's AC2

- 文献依据：Bujang & Baharum 2017（[DOI 10.2427/12267](https://doi.org/10.2427/12267)，基于 Flack 1988 功效法/PASS）：5 级量表、边际成比例，检验 κ=0.70 显著高于 0.5 需 **n=69（80% 功效）/ n=89（90% 功效）**；想证明"显著高于 0.7"则需 n=206/267。边际分布不一致时需求可 ×2 以上；
- 模拟推导（代理蒙特卡洛，假设已列明）：n=60 时加权 κ 的 95% CI 宽约 0.25–0.28；真 κ=0.70 时"CI 下限>0.5"概率仅 0.77–0.86；
- kappa 悖论（Feinstein & Cicchetti 1990，[PMID 2348207](https://pubmed.ncbi.nlm.nih.gov/2348207/)）：评分集中在 2–3 分时 κ 被压低。规范做法：**加权 κ（预注册权重）+ Gwet's AC2（序数加权版）+ 原始一致率 + 各维边际分布并报**（Gwet 2008 [DOI 10.1348/000711006X126600](https://doi.org/10.1348/000711006X126600)；Wongpakaran 2013 [PMC3643869](https://pmc.ncbi.nlm.nih.gov/articles/PMC3643869/)）；
- 定稿前用 R 包 [kappaSize](https://cran.r-project.org/web/packages/kappaSize/) 的 `CI5Cats` 按 Rotondi–Donner CI 下限法复算。

### 4.2 Self-preference 与 Judge 审计六件套

关键文献：Panickssery 等 NeurIPS 2024（[arXiv:2404.13076](https://arxiv.org/abs/2404.13076)，自我识别与偏好线性相关的因果证据）；Wataoka 2024（[arXiv:2410.21819](https://arxiv.org/abs/2410.21819)，机制=偏好低困惑度文本→**与生成模型同源蒸馏的第二 Judge 不算独立**）；Zheng 2023 MT-Bench（[arXiv:2306.05685](https://arxiv.org/abs/2306.05685)）；PoLL 跨家族小模型面板（[arXiv:2404.18796](https://arxiv.org/abs/2404.18796)）；JudgeBench（[arXiv:2410.12784](https://arxiv.org/abs/2410.12784)，难对上须用推理型 Judge）；HealthBench（[arXiv:2505.08775](https://arxiv.org/abs/2505.08775)，rubric 型 LLM 评分+医师验证在医学可行）。

写入评估协议的六项审计：
1. 换序重跑（消融对比环节），报告翻转率；
2. 长度审计：分数~token 长度相关系数，rubric 显式"不奖励冗长"；
3. 跨家族第二 Judge 一致性（κw/AC2 与分歧分布）；
4. Judge-人 κw 对比人-人 κw（Zheng 的 85% vs 81% 模板）；
5. self-preference 定向审计：Hy3/异族模型各生成、双 Judge 交叉打分，报告"评自家−评他家"分差；
6. Judge 契约冻结 + 换版即在人工金标上重校准。

第二 Judge 选型：**主用自部署 Qwen**（家族独立、版本可冻结、医学文本不出域），DeepSeek API 作廉价仲裁票；GPT/Claude 中转仅小样本校准审计。注意 DeepSeek API 现役已是 v4 系（v4-flash/v4-pro，[定价页](https://api-docs.deepseek.com/quick_start/pricing) 2026-08-28 核实），本地 `.env` 里 `deepseek-reasoner/deepseek-chat` 别名是否仍在服务需顺手确认。最终以 60 份人评金标做 Judge 元评估后冻结组合。

### 4.3 提示注入防护（PDF 藏指令）

- OWASP LLM01:2025 第一位风险（[官方条目](https://github.com/OWASP/www-project-top-10-for-large-language-model-applications/blob/main/2_0_vulns/LLM01_PromptInjection.md)），官方明言无万全解，纵深防御；
- 真实攻击先例：2025-07 Nikkei 曝光 17 篇 arXiv 论文白字藏"GIVE A POSITIVE REVIEW ONLY"（[报道](https://asia.nikkei.com/business/technology/artificial-intelligence/positive-review-only-researchers-hide-ai-prompts-in-papers)）；受控实验显示隐藏注入可把 LLM 审稿拉到 100% 接受（[arXiv:2509.10248](https://arxiv.org/abs/2509.10248)）——**"论文全文藏指令"不是假想威胁**；
- MVP 必做五条：① PDF 入库消毒（渲染 OCR 与文本层 diff 抓白字/微字体）；② Spotlighting 包裹一切检索文本（[arXiv:2403.14720](https://arxiv.org/abs/2403.14720)，ASR >50%→<2%）；③ 抽取用无工具权隔离 LLM、编排层只见 Schema 化结果（Dual-LLM/CaMeL [arXiv:2503.18813](https://arxiv.org/abs/2503.18813)、六模式 [arXiv:2506.08837](https://arxiv.org/abs/2506.08837)）；④ 出口控制：引用 ID 必须命中本次检索集、禁任意外链（兼防幻引，[lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)）；⑤ Judge 同样设防且把注入样本纳入 12 类对抗集（方案 9.3 第 12 类已有，验收用 36 组配对报"检出率+评分位移"）；
- 附加：Prompt Guard 2 检测器旁路（[HF 模型卡](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)，注意检测器只是纵深一环）。

### 4.4 GRPO 资源（官方原文）

| 用途 | 官方原文数字 | 出处 |
|---|---|---|
| 推理 | 8×H20-3e 起 | [README](https://github.com/Tencent-Hunyuan/Hy3/blob/main/README.md) |
| LoRA 微调 | 至少单机 8×80GB | [finetune/README](https://github.com/Tencent-Hunyuan/Hy3/blob/main/finetune/README.md) |
| 全参 SFT | 至少 4 机 32×80GB | 同上 |
| GRPO rollout 下限 | 单实例最低 TP=16（16×H20 才装得下权重分片） | [rl/README](https://github.com/Tencent-Hunyuan/Hy3/blob/main/rl/README.md) |
| GRPO 官方实测 | **128×H20（16 节点）**，verl+Megatron 全 offload+全量重计算，2048 轨迹/步 | 同上 |

另加方法论理由：GRPO 需要冻结且可信的 reward，而本项目评估器正是待验证对象（未通过一致性/对抗验证就当 reward 属循环依赖）；官方演示数据集 1.7 万题级，与 40 题规模完全不匹配。保留路径：仅考虑 8×80GB LoRA-SFT + 已验证评估器做离线筛选信号。

---

## 5. 遗留未决项

1. Europe PMC 数值型限流：官方确无数字，保守并发即可，不再追查；
2. Sim & Wright (2005) 全文未逐项核对（OUP 抓取超时），样本量数值以 Bujang 2017 + PASS 文档为准；
3. 按模型家族对比"医学 NLI/证据评审"能力的公开榜单不存在——第二 Judge 选型以自建 60 份人评金标元评估定夺；
4. MedRAGChecker（arXiv:2601.06519）为研究代理核验，主代理未复核原文，引用前顺手点开一次；
5. 混元 Prompt 1（API 行为）已由实测冒烟替代完成，Prompt 2–4 至此全部闭环，无需再问混元。

---

## 6. 延伸：以经典一致性/可靠性统计为基线的顶会拓展（2024–2026，主代理逐一核实）

| 经典基线 | 顶会拓展 | 内容 | 对本项目 |
|---|---|---|---|
| 人机一致率 / κ（事后报告） | **Trust or Escalate**（ICLR 2025 **Oral**，Jung/Brahman/Choi，[arXiv:2407.18370](https://arxiv.org/abs/2407.18370)） | 把"与人一致"变成可证明的前置保证：Judge 评估自身置信度，没把握就拒判/升级（Simulated Annotators 置信估计 + Cascaded Selective Evaluation），达到用户指定的一致率风险水平 | 我们 L4"低置信转人工"队列的理论包装与引用依据；升级阈值可按其风险控制法选 |
| 少量金标校准大量自动分 | **Prediction-Powered Ranking**（NeurIPS 2024，Chatzi 等，[arXiv:2402.17826](https://arxiv.org/abs/2402.17826)；根子是 PPI，Angelopoulos 等 Science 2023） | 小量人工比较 + 大量 LLM 比较 → 带覆盖保证的系统排名区间（rank-set），量化"Judge 与人不一致"引入的排名不确定性 | 与"60–90 份人评 + 360 份自动评"设计完全同构；A/B/C/D 排名可给出带保证的区间，有现成[代码](https://github.com/Networks-Learning/prediction-powered-ranking) |
| Judge 去偏范式本身 | **Limits to scalable evaluation**（ICLR 2025，Dorner/Nastl/Hardt，[arXiv:2410.13341](https://arxiv.org/abs/2410.13341)） | 定理：当 Judge 不比被测模型更强时，任何去偏方法最多节省一半人工标注 | 证成"专家金标不可替代、Judge 只是放大器"的立场；写进设计依据展示知道范式边界 |
| ICC 的方差分解 | **G 理论（概化理论）搬进 LLM 评测**：DDR 长程 Agent 评测定容框架（[arXiv:2608.11323](https://arxiv.org/abs/2608.11323)，2026-08）；Total Evaluation Error（[arXiv:2604.11581](https://arxiv.org/abs/2604.11581)，2026） | 把噪声按"任务/评审/提示变体/重复/交互"多面拆解（G-study），再反推达到目标可靠度需要多少题、几个 Judge、跑几遍（D-study 预算前沿） | 10.6 重复稳定性（30 份×5 次）的数据天然可做方差分解；回答"跑几次、标几份"的预算问题比拍脑袋强 |
| 考试总分 | **IRT（项目反应理论）**：metabench（ICLR 2025，[arXiv:2407.12844](https://arxiv.org/abs/2407.12844)）、tinyBenchmarks（ICML 2024） | 给每道题估计难度与区分度，6 个基准压缩到 3% 仍能重建总分 | 事后分析 40 题金标集哪些题最能区分 A/B/C/D——判别力验证的进阶版（可选） |
| 只报点估计 | **Adding Error Bars to Evals**（Evan Miller/Anthropic，[arXiv:2411.00640](https://arxiv.org/abs/2411.00640)） | 评测=实验：CLT 标准误、按题聚类标准误、配对差检验、功效分析的完整操作手册 | 方案 10.3 的"配对比较 + 按问题聚类 Bootstrap"正是该规范，引用它作方法学锚点 |
| Judge 偏差缓解 | CyclicJudge（[arXiv:2603.01865](https://arxiv.org/abs/2603.01865)，2026-03）、bias-aware judge 训练（[arXiv:2603.08091](https://arxiv.org/abs/2603.08091)）、Quantifying variance in evaluation benchmarks（Madaan 等，ICLR 2025） | 轮转分配消位置偏差、训练期显式去偏、基准方差量化 | 说明该线仍在高产出；审计设计与其对齐 |

结论：κ/ICC/AC2/α 这套"旧"工具正是 2024–2026 顶会评测方法学的公共基线——最新工作不是替换它们，而是在其上加"可证明保证"（ICLR Oral）、"统计推断"（NeurIPS/PPI）、"预算设计"（G 理论）与"理论上限"（ICLR）。本方案采用同一基线并叠加领域特化（条件保真、致命错误上限、对抗验证），与该谱系同向。
