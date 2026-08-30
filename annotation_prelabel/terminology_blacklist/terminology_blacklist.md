# 术语与表述误用黑名单（AI 预标）

> ## ⚠️ 本文件为 AI 预标草稿，未经人工核验，不得作为金标使用，不得计入双人一致性统计
>
> 标注者：`claude-fable-5-thinking (AI预标)` ｜ 生成日期：2026-08-28 ｜ 状态：`ai_prelabel_pending_human`
> 机器可读版：`terminology_blacklist.jsonl`（60 条记录）

---

## 0. 这份清单怎么来的

60 条中有 **38 条锚定了本地知识库中实际观察到的缺陷**（`observed_in_local_corpus` 字段给出具体 `statement_id`），不是凭空罗列的「常见错误」。数据来源是 `agent/knowledge/kg_edges.jsonl`（2,312 条边，sha256 `a23602c9…`）中通过 `_is_citable_claim` 结构过滤的 512 条可引用候选。

另外 22 条来自 β 细胞线粒体领域的方法学与命名常识，`ai_confidence` 标注了我的自评置信度，其中 5 条为 `medium` 并在 `needs_human_verification` 中明确要求人工确认来源后才可作为评分项。

## 1. 分布概览

| 类别 | 条数 | 主要用途 |
|---|---:|---|
| 方法学能力边界 | 9 | D4/D8：方法能测什么、不能测什么 |
| 术语规范 | 9 | D8：术语与条件表述规范 |
| 证据类型 | 8 | D1/D2/D5：证据来源资格 |
| 因果越界 | 6 | D5：相关/因果、必要/充分、模型/实测 |
| 方向与极性 | 6 | D4：效应方向（含致命错误类） |
| 统计与重复 | 5 | D3/D5：独立性与计数 |
| 物种细胞外推 | 4 | D4：物种偷换（致命错误类） |
| 命名与别名 | 4 | D3/D8：检索召回与去重 |
| 安全边界 | 4 | D6：拒答校准 |
| 文本损坏 | 3 | D7/D8：文本规范化与锚点定位 |
| 单位与数值 | 2 | D8：单位与量纲 |

**建议检出主体**：规则层 43 条、人工 11 条、语义 Judge 6 条。规则层占比高是有意的——核验报告 1.3 节引用的 CiteGuard（ACL 2026）实测「裸 LLM Judge 做引用判断 recall 仅 16–17%」，因此凡能规则化的都不交给 Judge。

**映射到九维的分布**：D4（28）与 D8（28）最密，D5（21）、D2（13）次之。这与方案把 D4「实验条件与效应方向准确性」列为关键维度一致。

---

## 2. 由本地数据确证的九类高优先级缺陷

这一节是本清单最有操作价值的部分：每一条都在真实数据中量化过，可以直接转成修复工单。

### 2.1 参考文献列表条目被当作正文证据 —— 41/512（8.0%）｜TERM-041

512 条结构可引用 Claim 中，**41 条的证据片段其实是 J Physiol 2000 综述参考文献列表里的论文标题**。全部被 section 分类器误标为 `conclusion`（该论文 49 条 `conclusion` 类 Claim 中 41 条属此类）。

典型例子：

| Claim | 证据片段（实为参考文献标题） |
|---|---|
| `glucose --regulates--> glutaminolysis` | 「Glucose regulation of glutaminolysis and its role in insulin secretion.」 |
| `mutant glutamate dehydrogenase --causes--> Hyperinsulinism-hyperammonemia syndrome` | 「Hyperinsulinism-hyperammonemia syndrome caused by mutant glutamate dehydrogenase」 |
| `Malonyl-CoA --mediates--> nutrient-induced insulin secretion` | 「Malonyl-CoA and long chain acyl-CoA esters as metabolic coupling factors in nutrient-induced insulin secretion.」 |

这类记录只能证明「存在一篇叫这个名字的论文」，不能支撑任何科学结论。**注意第 2 行的科学内容其实是正确的**（激活型 GDH 突变确实致 HI/HA 综合征）——但证据来源错误，仍须拒绝。这正是 D1（引用真实性）与 D2（主张—证据一致性）必须分开评分的实例。

诊断依据：该论文全文末尾仍完整保留参考文献段（起点约在全文 60.4% 处，可见 "Weaver, C. D., Yao, T. L., … Journal of Biological Chemistry 271, 12977—12984." 等条目），而另外三篇论文的参考文献段已被正确排除（污染 0 条）。**须查明清洗管线为何对该论文失效**（其 `fulltext_cleaning_segments=8`）。

### 2.2 模型模拟输出与实验测量不可区分 —— 131/512（25.6%）｜TERM-005

来源论文之一是 bioRxiv 2021 预印本，全文 `ODE` 出现 189 次、`model` 124 次、`simulat*` 31 次、`in silico` 14 次，而 `measured` 仅 1 次。其摘要明确写道模型「revealed that mitochondrial fission occurs in response to hyperglycemia, starvation, ATP synthase inhibition, uncoupling, and diabetic condition」。

也就是说，**该论文最有价值的几条结论恰恰是模型输出**。但其 131 条 Claim 中只有 19 条（14.5%）的证据片段字面含建模措辞，其余无法从片段本身判断是模拟还是实测。当前 Claim Schema 没有 `study_design` 字段承载这一区分，风险直接传导到答案层。

受影响的重点条目包括 `stmt_427d3be6d8b72c12ea89caab`（ATP 合酶抑制 → Δψm 升高 + ATP/ADP 下降）——这条同时是方案 9.1 难例清单「膜电位升高但 ATP 未同步增加」的最佳实例，价值高但来源资格待定。

### 2.3 方向反转 —— 已确证 1 例，且检测机制从未触发｜TERM-015

`stmt_537ec0a779806603ffc9d63d` 记录为 `mitochondrial dynamics --regulates--> ATP synthesis`，而原文写的是：

> The model also demonstrated that mitochondrial dynamics **were regulated by** ATP synthesis and proton leakage under various metabolic conditions.

方向完全相反，且 `direction_corrected=false`。**512 条记录的 `direction_corrected` 全部为 `false`**，说明方向纠正机制从未触发过一次。被动语态（`were regulated by` / `is mediated by` / `is induced by`）的方向检测须加入规则层并对 512 条全量回扫。

按方案 8.3，核心主张的效应方向反转触发总分上限 69，属致命错误类。

### 2.4 relation 与 polarity 语义互相矛盾｜TERM-016

`stmt_1309588298bc4e24f6b4d36f` 记为 `GIP receptor --protects_against--> glucose intolerance`，同时 `polarity=negative`。两者含义相反，无法判定效应方向。而原文说的是「knock-out of the GIP receptor in the mouse leads to glucose intolerance」——正确形式应是 `GIP receptor knock-out --causes--> glucose intolerance`（`species=mouse`），「GIPR 是正常糖耐量所必需」是由此得出的**推断**，不是原文陈述。

### 2.5 阴性结果被记为正向极性｜TERM-013

`stmt_8a3fd77695029c924c2b4d45` 的证据是「We observed **no differences** in random fed blood glucose levels between male βMcu-KO and control animals at all ages examined」，却记 `polarity=positive`。

本地 512 条的 polarity 只用了 `positive`(453) 与 `negative`(59) 两个值，**`no_effect` / `mixed` / `unknown` 完全未使用**，尽管方案 6.1 的 `effect_direction` 枚举已包含它们。方案 9.1 难例清单明确要求覆盖阴性结果，而当前 Schema 用法把阴性结果记成了正向。

### 2.6 一句多 Claim 造成分母膨胀 —— 195/512（38.1%）｜TERM-051 / TERM-052

83 个证据句支撑了 195 条 Claim，最多一句支撑 6 条。最典型的一句：

> The inhibition of ATP synthase activity resulted in higher mitochondrial membrane potentials, a lower ATP/ADP ratio, and a more fragmented mitochondrial network.

它产生了 6 条 Claim，其中 3 条头实体写作 `inhibition of ATP synthase activity`、3 条写作 `ATP synthase inhibition` —— **3 对完全等价的近重复记录**。

按 8.1 节的原子主张定义，把这句拆成 3 条（三个不同结局）是**正确的**；问题在两处：(a) 同义改写的头实体未归一，造成 3 条冗余；(b) 证据覆盖率与支持率的分母若按 Claim 条数计算，会系统性虚增证据量。修复方向是共享 `evidence_id`、按证据去重计分。

### 2.7 文本损坏 —— 63/512（12.3%）｜TERM-054

含 ﬁ/ﬂ 连字、上标丢失（`Ca¥`、`K¤`）、希腊字母损坏（`á_ketoglutarate`、`â_cell`）、连字符变下划线（`INS_1`）。分布：J Physiol 2000 综述 43 条、Br J Cancer 19 条、预印本 1 条。

危害有两层：直接触发 D8 术语/数字准确性扣分；更关键的是**破坏证据定位**——核验报告 3.1 节实测 Europe PMC Annotations API **没有字符 offset**，定位完全依赖 prefix/exact/postfix 文本精确匹配，片段里的损坏字符会让锚点直接失配。规范化必须在 `evidence_text` 与全文快照两侧同步执行，并重算 `full_text_sha256`。

### 2.8 证据片段过短或为残句｜TERM-055

`stmt_af826610aa5a40871c174fd7` 的 `evidence_text` 是 **「dose-dependent inhibition」**——3 个词，无主语无宾语，却 `evidence_match=true`、`confidence=0.95`。这类片段无法支持任何主张—证据判定，但完全通过了当前的 `_is_citable_claim` 结构过滤。

`_is_citable_claim` 目前只检查 `evidence_text` 非空，**没有最小长度或完整性判据**。

### 2.9 实体名夹带条件、程度副词或状态限定｜TERM-056

| 现状 | 问题 |
|---|---|
| head = `mitochondria at 0X` | 实体夹带剂量标签，且 `0X` 非量纲值 |
| tail = `moderately suppressed mitochondrial membrane potential` | 夹带程度副词 |
| tail = `normal fed blood glucose levels` | 夹带状态限定 |

这些都会让实体归一与跨论文对齐失败，进而破坏 D3 的证据覆盖统计。

---

## 3. 领域方法学条目（未由本地数据确证，须人工核实来源）

这 5 条 `ai_confidence=medium` 的条目全部来自我的领域知识而非池内文献，**在人工确认来源前不得作为评分项**：

| ID | 内容 | 待核 |
|---|---|---|
| TERM-018 | 阳离子探针淬灭/非淬灭模式下信号方向可相反 | 是否为池内综述明确陈述 |
| TERM-023 | 完整胰岛存在核心氧弥散梯度，与单细胞结果不可互换 | 同上 |
| TERM-025 | R-GECO 类红移钙指示剂对 pH 敏感 | 原始研究是否已讨论 |
| TERM-047 | 丙二酰辅酶 A 偶联因子假说受质疑；UCP2 主导地位存争议 | 须专家给出具体反证文献 |
| TERM-060 | 多轮追问下的拒答坚持性 | 当前单轮评测无法覆盖 |

TERM-019（NAD(P)H 自发荧光无法区分 NADH 与 NADPH）我给了 `high`，因为这是光谱学事实而非文献主张；但它牵连本地 bootstrap 用例 `citation-nadph-flim`，须确认该用例的证据片段是否正确写作 `NAD(P)H`。

---

## 4. 与既有 bootstrap 检索用例的冲突（须优先处理）

本地 15 条 bootstrap 检索用例中，至少 3 条的期望答案指向我判定为应拒绝或需修正的 Claim：

| 用例 | 依赖的 Claim | 我的判定 | 冲突 |
|---|---|---|---|
| `entity-drp1-local` | `stmt_0123d86e2f548b99b7c99328` | reject（乳腺癌论文 discussion 转述，无条件字段） | 回归用例锚定在跨域转述主张上 |
| `path-proton-leak-fission` | `stmt_e3d2a12978f14d533e58f40e` | reject（证据片段带他人引文标记 [6,51]） | 三跳路径的一步不成立 |
| `citation-insulin-beta-cell` | `stmt_b6b7731f4ae05b2a383989ce` | accept_with_edits（教科书级无信息量主张，condition 错配） | 回归用例锚定在无信息量主张上 |

这些用例已被标注 `not_scientific_gold: true`、`curation_status: bootstrap_needs_expert_review`，用作工程回归是正当的。但若 Claim 层按本清单修复，**用例期望值须同步修订**，否则回归会在修复后反而失败。

---

## 5. 字段说明

| 字段 | 含义 |
|---|---|
| `term_id` | 稳定编号 TERM-001 … TERM-060 |
| `category` | 11 类之一（见 §1） |
| `wrong` / `correct` | 误用表述 / 规范表述 |
| `why` | 为什么错，含本地数据的量化依据 |
| `example_wrong` / `example_correct` | 可直接用于对抗样本模板与 Judge 提示词的最小对照例 |
| `maps_to_dimension` | 命中的九维编号（D1—D9） |
| `detector` | 建议检出主体：`rule`（确定性规则）/ `judge`（语义 Judge）/ `human`（须人工） |
| `observed_in_local_corpus` | 在本地语料中确证该缺陷的 `statement_id` 列表，`null` 表示未在本地确证 |
| `ai_confidence` | AI 自评置信度，**不是**证据强度，不得用于评分 |
| `needs_human_verification` | 待人工核实项，核完应清空 |

## 6. 建议用法

1. **规则层实现**：43 条 `detector=rule` 的条目可直接转为检查函数。建议优先实现 §2 的九类（有量化基线，可测修复效果）。
2. **对抗样本模板**：`example_wrong` / `example_correct` 构成天然的单因素配对，可直接对接方案 9.3 的 12 类攻击（TERM-001→第 5 类因果改写、TERM-007→第 3 类物种偷换、TERM-015→第 4 类方向交换、TERM-006→第 11 类预测边写成事实、TERM-049→第 8 类同源数据误作独立）。
3. **Judge 提示词**：6 条 `detector=judge` 的条目应写入冻结提示词。特别注意 TERM-059 —— **须显式告知 Judge「免责声明不减轻越界判定」**，否则语义 Judge 可能因「请遵医嘱」而降低严重度评级。
4. **不要**把本清单当作完备集。它由一次 50 条抽样与 4 篇论文的语料推出，覆盖面受限于该语料；主池 12 篇综述的全文尚未纳入分析。
