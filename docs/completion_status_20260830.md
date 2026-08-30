# MitoEvidence-Hy3 完成状态与正式实验边界

> 快照日期：2026-08-30
> 结论：工程技术底座已从“评估器原型”推进到“可运行的应用与评估闭环 MVP”；
> 科学金标和正式赛题实验仍必须由真实 Hy3 新 Key、领域专家和冻结数据共同完成。

## 1. 已经完成并可复核的工程能力

| 工作线 | 已完成内容 | 可核查位置 |
|---|---|---|
| 应用闭环 | 研究问题、Hy3 检索计划、冻结全文 BM25、证据约束生成、Claim—EvidenceSpan 绑定、Judge 输入和哈希审计包 | `app/`、`scripts/run_review.py`、`scripts/run_pilot_suite.py` |
| 证据池 | 12 篇种子综述、2,043 条原始参考文献、1,623 条去重候选；7 篇 OA XML 按 SHA-256 冻结 | `eval/data/evidence_pool_manifest.json`、`eval/data/evidence_pool_candidates.jsonl` |
| 原文定位 | 安全解析 JATS XML；prefix/exact/postfix 与 section 联合重定位；输出 found/ambiguous/not_found | `tools/literature/xml_anchor.py` |
| 独立主张候选 | 只读取问题与答案，不接受被测系统自报 Claim；候选保存源句、字符偏移和风险标志 | `evaluator/claim_splitter.py` |
| 九维评估 | D1—D9 算术、NA、事件上限、PASS/REVIEW/REJECT、四类致命错误证据审计 | `evaluator/rubric.py`、`evaluator/assembly.py` |
| 语义 Judge | Hy3 Function Calling、本地 Schema、有限重试、自一致性置信和人工升级 | `evaluator/judge/` |
| 确定性规则 | DOI/PMID/PMCID、数字与单位、D7/D9 清单、本地术语三态初筛 | `evaluator/rules/` |
| 双专家入口 | AI 字段白名单隔离、中性盲标包、A/B 锁定校验、分歧清单、空白第三人裁决表 | `evaluator/blind.py` |
| 有效性统计 | 好/中/差判别力、Kendall tau-b、Cohen/加权 κ、Spearman、MAE、ICC(2,1)、稳定性和对抗性 | `evaluator/validation.py` |
| AI 预标材料 | 5 道 Pilot、50 条 Claim、60 条术语风险、12 篇综述池预评估；全部醒目标注为非金标 | `annotation_prelabel/` |

应用运行清单记录模型/端点来源、提示词/Schema/配置哈希、token 用量、语料快照、
源 XML 哈希和锚点状态；不保存 API Key 或 `reasoning_content`。

## 2. 已执行的工程验证

### 2.1 全量离线回归

- Python 3.11；
- 400 项测试全部通过；
- 在仓库外新建的空白虚拟环境中按 `requirements.lock.txt` 重新安装后，400 项再次全部通过；
- 测试不访问 Hy3、Crossref、NCBI 或 Europe PMC；
- HTTP、限流和模型响应均使用注入式 mock；
- `git diff --check` 通过。

冻结语料准备器另行核对 manifest 中 7 个 OA XML：本机 7/7 文件 SHA-256 一致；
全新克隆可运行 `scripts/fetch_frozen_corpus.py` 获取缺失文件。若 Europe PMC 返回的
字节与冻结哈希不同，工具只报告 snapshot drift，不会静默覆盖或改写 manifest。

### 2.2 五题 Pilot 中性输入回归

离线替身只用于检查编排与安全停止，不生成科学答案。五题结果：

| Pilot | 工程状态 | 召回段落 | 候选主张 | 锚点 |
|---|---|---:|---:|---|
| PILOT-01 | 完成 | 4 | 1 | found |
| PILOT-02 | 完成 | 4 | 1 | found |
| PILOT-03 | 完成 | 4 | 1 | found |
| PILOT-04 | 完成 | 4 | 1 | found |
| PILOT-05（越界） | 安全拒答 | 0 | 0 | 不适用 |

应用加载器只读取 `question_id`、`question` 和 `scope`；不会把 AI 预标的
`answerability`、`source_reviews`、`required_claims` 或 `prohibited_inferences` 泄露给被测应用。
因此该回归能证明输入隔离和工程链路，但**不能证明 Hy3 能力或科研结论正确性**。

### 2.3 盲标包审计

实际数据可生成 127 条中性记录：5 个 Pilot、50 个 Claim、60 个术语项和 12 个综述池项。
中性包不含 AI 决定、AI 理由、`required_claims`、`prohibited_inferences` 或其他会锚定专家的字段；
输入和输出均有 SHA-256。A/B 文件目前仍是空白模板。

## 3. 还没有完成，不能写进“实验结果”

以下工作依赖新的权限、真实专家或尚未冻结的数据，代码不能替代：

1. **吊销并轮换旧 Hy3 API Key。** 明文已从项目文档替换为 `${HY3_API_KEY}`，但控制台吊销必须由账号持有人执行；旧 Key 在完成吊销前应视为泄露。
2. **真实 Hy3 五题运行。** 只能在新 Key 生效后执行；离线 smoke 不得替代。
3. **双专家独立盲标与第三人裁决。** 当前不存在正式专家金标，也不存在可报告的专家 κ/ICC。
4. **20 份校准输出。** 完成后才能冻结 claim splitter、量表、Judge Prompt、Schema 和人工升级阈值。
5. **正式评测集。** 仍需扩展并冻结 40 题核心集、20 题拒答集和至少 36 组对抗配对，按论文/证据源隔离划分。
6. **正式实验。** 尚未运行 A（直接 Hy3）、B（向量 RAG）、C（证据图）、D-auto、D-reviewed 消融，也没有逐题结果表。
7. **完整术语权威对齐。** 当前本地词表仅做三态初筛，不是完整 MeSH/GO 镜像，不能直接生成 D8 正式准确率。
8. **2 分钟 Demo/GIF 与许可。** Demo 尚未录制；仓库许可证需要项目权利人作出法律选择后再添加。

## 4. 正式完成的推荐顺序

1. 在腾讯控制台吊销旧 Key，创建最小权限新 Key，只通过环境变量注入；
2. 对 5 道 Pilot 执行真实 Hy3 套件，检查每条 Claim 的证据锚点和失败运行；
3. 将 127 条中性包按任务拆成小批次，先完成 5 道 Pilot 与 20 份量表校准；
4. A/B 专家独立锁定，计算一致性，分歧交第三人裁决；
5. 冻结 splitter、Rubric、Judge、术语规则、问题集和语料 manifest；
6. 构造好/中/差、重复评分和对抗配对，运行 `scripts/analyze_validation.py`；
7. 执行 A/B/C/D-auto/D-reviewed，输出维度表、致命错误、置信区间和典型失败 Case；
8. 完成 README 发布版、开源许可证、结果表与 2 分钟演示。

## 5. 与赛题完成条件的对应

| 赛题要求 | 当前判断 |
|---|---|
| 真实用户与开放式应用 | 场景和应用 MVP 已完成；真实 Hy3 正式运行待新 Key |
| 5 个以上可操作维度 | 九维量表和自动汇总已完成；量表待专家校准冻结 |
| 自动/半自动评测流程 | 规则 + Hy3 Judge + 人工升级的代码链路已完成 |
| 难例与反例 | 设计和生成工具已具备；正式集合尚未冻结 |
| 判别力验证 | 统计脚本已完成；正式数据尚未运行 |
| 一致性验证 | 盲标与统计脚本已完成；专家标签尚未产生 |
| 对抗验证 | 指标和攻击口径已完成；正式配对尚未运行 |
| 完整结果表与 Case 分析 | 尚未完成 |
| 开源仓库与 2 分钟 Demo | 工程仓库已成形；许可证、发布清理和 Demo 待完成 |

因此，当前最准确的表述是：

> **MitoEvidence-Hy3 已完成可运行、可审计的应用与评估工程 MVP；评估方法的科学有效性
> 仍须通过新 Key 下的真实 Hy3 运行、双专家金标、量表冻结和正式实验来证明。**
