"use strict";

const DATA_URL = "./data/annotations.json";
const REPOSITORY_BLOB_URL = "https://github.com/blues-kun/hy-agent/blob/main/";

const FILTER_LABELS = {
  all: "选择类别后筛选",
  pilot_questions: "可回答性",
  claim_reviews: "准入结论",
  terminology_rules: "建议检测方式",
  review_pool: "全文状态",
};

const SEGMENT_COLORS = {
  answerable: "#34785d",
  partial: "#b07016",
  out_of_scope: "#a83f2c",
  accept: "#34785d",
  accept_with_edits: "#b07016",
  reject: "#a83f2c",
  uncertain: "#667083",
  rule: "#118c8b",
  judge: "#315ea8",
  human: "#7c4f9e",
  local_xml_verified_in_manifest: "#34785d",
  no_pmcid_xml_unavailable: "#b07016",
  manifest_xml_unavailable: "#a83f2c",
};

const QUICK_VIEWS = [
  {
    id: "claim_usable",
    dataset: "claim_reviews",
    icon: "✓",
    title: "可用于 β 细胞证据",
    description: "按独立可用性字段筛选，不等同于 accept",
    predicate: (record) => record.record.usable_for_beta_cell_evidence === true,
  },
  {
    id: "claim_attention",
    dataset: "claim_reviews",
    icon: "↺",
    title: "需要修改、拒绝或待定",
    description: "集中处理不能直接纳入的 Claim",
    predicate: (record) =>
      ["accept_with_edits", "reject", "uncertain"].includes(record.record.ai_decision),
  },
  {
    id: "term_human",
    dataset: "terminology_rules",
    icon: "◎",
    title: "建议人工判断的规则",
    description: "detector=human 是执行通道，不是标注来源",
    predicate: (record) => record.record.detector === "human",
  },
  {
    id: "review_fulltext",
    dataset: "review_pool",
    icon: "▤",
    title: "已有冻结全文的综述",
    description: "本地 XML 与 SHA-256 已写入 Manifest",
    predicate: (record) =>
      record.record.fulltext?.status === "local_xml_verified_in_manifest",
  },
];

const STATUS_LABELS = {
  answerable: "可回答",
  partial: "部分可回答",
  out_of_scope: "超出范围",
  accept: "接受",
  accept_with_edits: "修改后接受",
  reject: "拒绝",
  uncertain: "不确定",
  rule: "规则检测",
  judge: "语义评审",
  human: "人工判断",
  high: "高置信",
  medium: "中置信",
  low: "低置信",
  local_xml_verified_in_manifest: "本地 XML 已核验",
  manifest_xml_unavailable: "XML 不可用",
  no_pmcid_xml_unavailable: "无 PMCID / XML 不可用",
  abstract_only_no_pmcid: "仅摘要",
  abstract_only_no_oa_xml: "仅摘要 / 无 OA XML",
  unknown: "未记录",
  peer_reviewed_primary: "同行评议原始研究",
  review_secondary: "综述二级证据",
  off_domain_primary: "域外原始研究",
  preprint_computational_model: "预印本计算模型",
  results: "结果",
  discussion: "讨论",
  methods: "方法",
  abstract: "摘要",
  conclusion: "结论",
  front_matter: "前置信息",
};

const STATUS_TONES = {
  answerable: "positive",
  accept: "positive",
  local_xml_verified_in_manifest: "positive",
  rule: "positive",
  partial: "warning",
  accept_with_edits: "warning",
  uncertain: "warning",
  judge: "warning",
  manifest_xml_unavailable: "warning",
  no_pmcid_xml_unavailable: "warning",
  abstract_only_no_pmcid: "warning",
  abstract_only_no_oa_xml: "warning",
  out_of_scope: "danger",
  reject: "danger",
  human: "neutral",
  high: "positive",
  medium: "warning",
  low: "danger",
  unknown: "neutral",
};

const FIELD_LABELS = {
  species: "物种",
  tissue: "组织",
  cell_type: "细胞类型",
  intervention: "干预",
  dose: "剂量",
  time: "时间",
  effect_direction: "效应方向",
  method: "方法",
  glucose: "葡萄糖条件",
  model: "模型",
  sex: "性别",
  age: "年龄",
  action: "处理动作",
  evidence_text: "证据文本修订",
  triple: "三元组修订",
  relation: "关系修订",
  head: "头实体修订",
  tail: "尾实体修订",
  condition: "条件修订",
  certainty: "确定性修订",
  note: "备注",
  polarity: "极性修订",
  outcome: "结局修订",
};

const state = {
  payload: null,
  dataset: "all",
  status: "all",
  riskOnly: false,
  preset: null,
  sort: "id",
  query: "",
  selectedKey: null,
};

let fileHashWrittenByApp = null;

const dom = {
  metrics: document.querySelector("#metrics"),
  heroSummary: document.querySelector("#hero-summary"),
  flow: document.querySelector("#annotation-flow"),
  coverage: document.querySelector("#coverage-bars"),
  distributions: document.querySelector("#distribution-panels"),
  referenceTotal: document.querySelector("#reference-total"),
  dialogTotal: document.querySelector("#dialog-total"),
  quickViews: document.querySelector("#quick-views"),
  tabs: document.querySelector("#dataset-tabs"),
  activePreset: document.querySelector("#active-preset"),
  search: document.querySelector("#search-input"),
  status: document.querySelector("#status-filter"),
  risk: document.querySelector("#risk-filter"),
  sort: document.querySelector("#sort-order"),
  statusLabel: document.querySelector("#status-filter-label"),
  downloadJson: document.querySelector("#download-json"),
  exportCsv: document.querySelector("#export-csv"),
  copyViewLink: document.querySelector("#copy-view-link"),
  clear: document.querySelector("#clear-filters"),
  count: document.querySelector("#result-count"),
  context: document.querySelector("#result-context"),
  list: document.querySelector("#record-list"),
  detail: document.querySelector("#detail-pane"),
  date: document.querySelector("#snapshot-date"),
  hash: document.querySelector("#manifest-hash"),
  provenanceButton: document.querySelector("#provenance-more"),
  scopeNote: document.querySelector("#scope-note"),
  provenanceDialog: document.querySelector("#provenance-dialog"),
  manifestLink: document.querySelector("#manifest-link"),
  toast: document.querySelector("#toast"),
};

function element(tag, className, text) {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text !== undefined && text !== null) value.textContent = String(text);
  return value;
}

function append(parent, ...children) {
  for (const child of children.flat()) {
    if (child !== null && child !== undefined) parent.append(child);
  }
  return parent;
}

function safeTitleElement(tag, className, text) {
  const parent = element(tag, className);
  const tokens = String(text || "").split(/(<\/?i>)/i);
  let italic = null;
  tokens.forEach((token) => {
    if (/^<i>$/i.test(token)) {
      italic = element("em", "");
      parent.append(italic);
    } else if (/^<\/i>$/i.test(token)) {
      italic = null;
    } else if (token) {
      (italic || parent).append(document.createTextNode(token));
    }
  });
  return parent;
}

function hasValue(value) {
  if (value === null || value === undefined) return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return String(value).trim().length > 0;
}

function displayValue(value, emptyLabel = "未记录") {
  if (value === null || value === undefined) return emptyLabel;
  if (typeof value === "boolean") return value ? "是" : "否";
  if (Array.isArray(value)) {
    return value.length ? value.map((item) => displayValue(item, emptyLabel)).join("、") : emptyLabel;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value);
    return entries.length
      ? entries.map(([key, item]) => `${FIELD_LABELS[key] || key}: ${displayValue(item)}`).join("；")
      : emptyLabel;
  }
  const text = String(value).trim();
  return text || emptyLabel;
}

function labelStatus(status) {
  return STATUS_LABELS[status] || status || STATUS_LABELS.unknown;
}

function formatCode(value) {
  if (value === null || value === undefined) return "未记录";
  return STATUS_LABELS[value] || String(value).replaceAll("_", " ");
}

function statusPill(status) {
  const pill = element("span", "status-pill", labelStatus(status));
  pill.dataset.tone = STATUS_TONES[status] || "neutral";
  return pill;
}

function confidencePill(confidence) {
  return statusPill(confidence || "unknown");
}

function chip(text, tone = "default") {
  const value = element("span", "chip", text);
  if (tone === "risk") value.classList.add("chip--risk");
  if (tone === "neutral") value.classList.add("chip--neutral");
  return value;
}

function chipRow(values, tone = "default", emptyLabel = "未记录") {
  const row = element("div", "chip-row");
  if (!hasValue(values)) {
    row.append(chip(emptyLabel, "neutral"));
    return row;
  }
  const list = Array.isArray(values) ? values : [values];
  list.forEach((value) => row.append(chip(displayValue(value), tone)));
  return row;
}

function section(title, hint, ...children) {
  const wrapper = element("section", "detail-section");
  const heading = element("div", "section-heading");
  append(heading, element("h3", "", title));
  if (hint) append(heading, element("span", "", hint));
  append(wrapper, heading, children);
  return wrapper;
}

function paragraph(text, className = "body-copy") {
  return element("p", className, displayValue(text));
}

function factGrid(facts) {
  const grid = element("div", "fact-grid");
  for (const [label, value] of facts) {
    const card = element("div", "fact-card");
    append(card, element("span", "", label), element("strong", "", displayValue(value)));
    grid.append(card);
  }
  return grid;
}

function listCard(values, emptyLabel = "暂无记录") {
  if (!hasValue(values)) return notice(emptyLabel, "neutral");
  const list = element("ol", "list-card");
  values.forEach((value, index) => {
    const item = element("li", "", displayValue(value));
    item.dataset.index = String(index + 1).padStart(2, "0");
    list.append(item);
  });
  return list;
}

function notice(text, tone = "neutral") {
  const value = element("div", `notice-card notice-card--${tone}`);
  value.append(element("span", "", text));
  return value;
}

function objectCards(value, emptyLabel = "未记录结构化内容") {
  if (!hasValue(value)) return notice(emptyLabel, "neutral");
  const grid = element("div", "fact-grid");
  Object.entries(value).forEach(([key, item]) => {
    const card = element("div", "fact-card");
    append(
      card,
      element("span", "", FIELD_LABELS[key] || key),
      element("strong", "", displayValue(item)),
    );
    grid.append(card);
  });
  return grid;
}

function quote(text) {
  return element("blockquote", "quote-card", displayValue(text));
}

function compareCards(leftLabel, leftValue, rightLabel, rightValue) {
  const grid = element("div", "compare-grid");
  const left = element("div", "compare-card compare-card--wrong");
  const right = element("div", "compare-card compare-card--right");
  append(left, element("span", "compare-card__label", leftLabel), paragraph(leftValue));
  append(right, element("span", "compare-card__label", rightLabel), paragraph(rightValue));
  append(grid, left, right);
  return grid;
}

function formatDate(value) {
  if (!value) return "日期未记录";
  const parts = String(value).split("-");
  if (parts.length !== 3) return String(value);
  return `${parts[0]} 年 ${Number(parts[1])} 月 ${Number(parts[2])} 日确认`;
}

function getDataset(name) {
  return state.payload.datasets.find((dataset) => dataset.name === name);
}

function allRecords() {
  return state.payload.datasets.flatMap((dataset) =>
    dataset.records.map((record) => ({ ...record, datasetMeta: dataset })),
  );
}

function recordKey(record) {
  return `${record.dataset}/${record.id}`;
}

function parseHash() {
  try {
    const raw = window.location.hash.replace(/^#/, "");
    const slash = raw.indexOf("/");
    if (slash < 1) return null;
    return {
      dataset: decodeURIComponent(raw.slice(0, slash)),
      id: decodeURIComponent(raw.slice(slash + 1)),
    };
  } catch (_error) {
    return null;
  }
}

function updateHash(record, push = false) {
  const next = `#${encodeURIComponent(record.dataset)}/${encodeURIComponent(record.id)}`;
  if (window.location.hash === next) return;
  try {
    if (window.location.protocol === "file:") {
      fileHashWrittenByApp = next;
      window.location.hash = next;
    } else if (push) {
      history.pushState(null, "", next);
    } else {
      history.replaceState(null, "", next);
    }
  } catch (_error) {
    try {
      if (window.location.protocol === "file:") fileHashWrittenByApp = next;
      window.location.hash = next;
    } catch (_fallbackError) {
      // Navigation metadata must never prevent the verified data from rendering.
    }
  }
}

function clearHash() {
  if (!window.location.hash) return;
  try {
    if (window.location.protocol === "file:") {
      fileHashWrittenByApp = "";
      window.location.hash = "";
    } else {
      history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
    }
  } catch (_error) {
    try {
      if (window.location.protocol === "file:") fileHashWrittenByApp = "";
      window.location.hash = "";
    } catch (_fallbackError) {
      // A failed URL cleanup is cosmetic and must not break the review surface.
    }
  }
}

function normalizeQuery(value) {
  return String(value || "").normalize("NFKC").toLocaleLowerCase("zh-CN");
}

function getQuickView(id) {
  return QUICK_VIEWS.find((view) => view.id === id) || null;
}

function compareRecordIds(left, right) {
  return String(left.id).localeCompare(String(right.id), "zh-CN", {
    numeric: true,
    sensitivity: "base",
  });
}

function filteredRecords() {
  const query = normalizeQuery(state.query.trim());
  const preset = getQuickView(state.preset);
  const records = allRecords().filter((record) => {
    if (state.dataset !== "all" && record.dataset !== state.dataset) return false;
    if (state.status !== "all" && record.status !== state.status) return false;
    if (state.riskOnly && !record.risk_flags.length) return false;
    if (preset && !preset.predicate(record)) return false;
    if (query && !record.search_text.includes(query)) return false;
    return true;
  });
  return records.sort((left, right) => {
    if (state.sort === "risk") {
      return right.risk_flags.length - left.risk_flags.length || compareRecordIds(left, right);
    }
    if (state.sort === "title") {
      return left.title.localeCompare(right.title, "zh-CN") || compareRecordIds(left, right);
    }
    return compareRecordIds(left, right);
  });
}

function renderMetrics() {
  const purpose = {
    pilot_questions: "定义问题范围、必需主张和禁止推断",
    claim_reviews: "决定候选证据接受、修改或拒绝",
    terminology_rules: "统一专业表达并拦截常见科研越界",
    review_pool: "组织检索种子、全文状态和使用限制",
  };
  const cards = [];
  const total = element("button", "metric metric--total");
  total.type = "button";
  append(
    total,
    element("span", "", "全部参考记录"),
    element("strong", "", state.payload.summary.total_records),
    element("small", "", "4 类 · 单一汇总专家结果"),
    element("p", "metric__purpose", "从研究问题到文献来源的完整参考快照"),
  );
  total.addEventListener("click", () => chooseDataset("all"));
  cards.push(total);
  const accentClasses = ["metric--pilot", "metric--claim", "metric--term", "metric--review"];
  state.payload.datasets.forEach((dataset, index) => {
    const card = element("button", `metric ${accentClasses[index] || ""}`);
    card.type = "button";
    append(
      card,
      element("span", "", dataset.label),
      element("strong", "", dataset.record_count),
      element("small", "", dataset.description),
      element("p", "metric__purpose", purpose[dataset.name]),
    );
    card.addEventListener("click", () => chooseDataset(dataset.name));
    cards.push(card);
  });
  dom.metrics.replaceChildren(...cards);
}

function renderFlow() {
  const descriptions = {
    pilot_questions: ["研究问题", "先定义怎样才算答对"],
    claim_reviews: ["证据主张", "再判断哪些证据可采用"],
    terminology_rules: ["写作规则", "随后约束科研表达边界"],
    review_pool: ["文献来源", "最后回到综述与原始研究"],
  };
  const steps = state.payload.datasets.map((dataset, index) => {
    const button = element("button", "flow-step");
    button.type = "button";
    button.setAttribute("aria-label", `查看 ${dataset.label} ${dataset.record_count} 条`);
    const copy = element("span", "");
    append(
      copy,
      element("strong", "", descriptions[dataset.name][0]),
      element("small", "", descriptions[dataset.name][1]),
    );
    append(
      button,
      element("span", "flow-step__number", String(index + 1).padStart(2, "0")),
      copy,
      element("span", "flow-step__count", dataset.record_count),
    );
    button.addEventListener("click", () => chooseDataset(dataset.name));
    return button;
  });
  dom.flow.replaceChildren(...steps);
  const labels = Object.fromEntries(
    state.payload.datasets.map((dataset) => [dataset.name, dataset.record_count]),
  );
  dom.heroSummary.textContent =
    `${state.payload.summary.total_records} 条人工确认的汇总参考：` +
    `${labels.pilot_questions} 个研究问题、${labels.claim_reviews} 条证据主张、` +
    `${labels.terminology_rules} 条写作规则和 ${labels.review_pool} 篇种子综述。`;
  dom.referenceTotal.textContent = state.payload.analytics.review_pool.reference_count_total.toLocaleString("zh-CN");
  dom.dialogTotal.textContent = state.payload.summary.total_records;
}

function coverageRow(label, value, total, note, tone = "default") {
  const row = element("div", "coverage-row");
  row.dataset.tone = tone;
  const meta = element("div", "coverage-row__meta");
  append(meta, element("strong", "", label), element("span", "", `${value}/${total}`));
  const track = element("div", "coverage-track");
  track.setAttribute("role", "progressbar");
  track.setAttribute("aria-label", `${label} ${value}/${total}`);
  track.setAttribute("aria-valuemin", "0");
  track.setAttribute("aria-valuemax", String(total));
  track.setAttribute("aria-valuenow", String(value));
  const fill = element("span", "");
  fill.style.setProperty("--coverage", `${total ? (value / total) * 100 : 0}%`);
  track.append(fill);
  append(row, meta, track, element("small", "", note));
  return row;
}

function renderCoverage() {
  const analytics = state.payload.analytics;
  const pilot = analytics.pilot_questions;
  const claims = analytics.claim_reviews;
  const terms = analytics.terminology_rules;
  const reviews = analytics.review_pool;
  const pilotTotal = getDataset("pilot_questions").record_count;
  const claimTotal = getDataset("claim_reviews").record_count;
  const termTotal = getDataset("terminology_rules").record_count;
  const reviewTotal = getDataset("review_pool").record_count;
  dom.coverage.replaceChildren(
    coverageRow(
      "Pilot 原文证据锚点",
      Math.min(pilot.questions_with_evidence_papers, pilot.questions_with_evidence_spans),
      pilotTotal,
      `${pilot.resolvable_source_review_links}/${pilot.source_review_links} 个综述关联可跨表跳转；但综述线索不等于原文锚点`,
      "danger",
    ),
    coverageRow(
      "Claim 已记录实验条件",
      claims.with_recorded_conditions,
      claimTotal,
      `另有 ${claims.without_recorded_conditions} 条条件字段为空`,
      "warning",
    ),
    coverageRow(
      "术语规则有本地实例",
      terms.local_corpus_checked,
      termTotal,
      `共登记 ${terms.local_statement_links} 次 statement 引用，其中 ${terms.resolvable_local_statement_links} 次可跳转 Claim`,
      "warning",
    ),
    coverageRow(
      "综述已冻结本地全文",
      reviews.local_xml_verified,
      reviewTotal,
      `PMCID 已记录 ${reviews.with_pmcid}/${reviewTotal}`,
      "warning",
    ),
  );
}

function distributionRow(label, values) {
  const row = element("div", "distribution-row");
  const total = Object.values(values).reduce((sum, value) => sum + Number(value), 0);
  const meta = element("div", "distribution-row__meta");
  append(meta, element("strong", "", label), element("span", "", `${total} 条`));
  const bar = element("div", "stacked-bar");
  bar.setAttribute("aria-label", `${label}分布`);
  const legend = element("div", "distribution-legend");
  Object.entries(values).forEach(([status, count]) => {
    const color = SEGMENT_COLORS[status] || "#667083";
    const segment = element("span", "");
    segment.style.setProperty("--segment", `${total ? (count / total) * 100 : 0}%`);
    segment.style.setProperty("--segment-color", color);
    segment.title = `${labelStatus(status)} ${count}`;
    bar.append(segment);
    const item = element("span", "legend-item", `${labelStatus(status)} ${count}`);
    item.style.setProperty("--legend-color", color);
    legend.append(item);
  });
  append(row, meta, bar, legend);
  return row;
}

function renderDistributions() {
  const analytics = state.payload.analytics;
  const reviewStatus = analytics.review_pool.fulltext_status;
  const reviewGrouped = {
    local_xml_verified_in_manifest: reviewStatus.local_xml_verified_in_manifest || 0,
    manifest_xml_unavailable:
      (reviewStatus.manifest_xml_unavailable || 0) +
      (reviewStatus.no_pmcid_xml_unavailable || 0),
  };
  dom.distributions.replaceChildren(
    distributionRow("Pilot 可回答性", analytics.pilot_questions.answerability),
    distributionRow("Claim 准入结论", analytics.claim_reviews.decision),
    distributionRow("术语建议检测方式", analytics.terminology_rules.detector),
    distributionRow("综述全文状态", reviewGrouped),
  );
}

function renderQuickViews() {
  const records = allRecords();
  const cards = QUICK_VIEWS.map((view) => {
    const count = records.filter(view.predicate).length;
    const button = element("button", "quick-view");
    button.type = "button";
    button.dataset.preset = view.id;
    button.setAttribute("aria-pressed", String(state.preset === view.id));
    const top = element("span", "quick-view__top");
    append(
      top,
      element("span", "quick-view__icon", view.icon),
      element("span", "quick-view__count", count),
    );
    append(
      button,
      top,
      element("strong", "", view.title),
      element("small", "", view.description),
    );
    button.addEventListener("click", () => applyQuickView(view.id));
    return button;
  });
  dom.quickViews.replaceChildren(...cards);
}

function applyQuickView(id) {
  if (!state.payload) return;
  const view = getQuickView(id);
  if (!view) return;
  state.preset = state.preset === id ? null : id;
  state.dataset = state.preset ? view.dataset : "all";
  state.status = "all";
  state.riskOnly = false;
  state.sort = "id";
  state.query = "";
  state.selectedKey = null;
  document.body.classList.remove("mobile-detail-open");
  dom.search.value = "";
  dom.risk.checked = false;
  dom.sort.value = "id";
  renderQuickViews();
  renderTabs();
  renderStatusOptions();
  renderRecords();
  document.querySelector("#review")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function chooseDataset(name) {
  if (!state.payload) return;
  state.dataset = name;
  state.status = "all";
  state.preset = null;
  state.selectedKey = null;
  document.body.classList.remove("mobile-detail-open");
  renderQuickViews();
  renderTabs();
  renderStatusOptions();
  renderRecords();
  document.querySelector(".review-surface")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderTabs() {
  const definitions = [
    { name: "all", label: "全部", count: state.payload.summary.total_records },
    ...state.payload.datasets.map((dataset) => ({
      name: dataset.name,
      label: dataset.label,
      count: dataset.record_count,
    })),
  ];
  const tabs = definitions.map((definition) => {
    const button = element("button", "dataset-tab");
    button.type = "button";
    button.setAttribute("role", "tab");
    button.dataset.dataset = definition.name;
    button.setAttribute("aria-selected", String(state.dataset === definition.name));
    append(button, document.createTextNode(definition.label), element("span", "", definition.count));
    button.addEventListener("click", () => {
      chooseDataset(definition.name);
    });
    return button;
  });
  dom.tabs.replaceChildren(...tabs);
}

function renderStatusOptions() {
  const records = allRecords().filter(
    (record) => state.dataset === "all" || record.dataset === state.dataset,
  );
  const statuses = [...new Set(records.map((record) => record.status))].sort((a, b) =>
    labelStatus(a).localeCompare(labelStatus(b), "zh-CN"),
  );
  if (state.status !== "all" && !statuses.includes(state.status)) state.status = "all";
  const allView = state.dataset === "all";
  dom.statusLabel.textContent = FILTER_LABELS[state.dataset] || "状态";
  const options = [new Option(allView ? "选择类别后筛选" : "全部状态", "all")];
  statuses.forEach((status) => options.push(new Option(labelStatus(status), status)));
  dom.status.replaceChildren(...options);
  dom.status.value = state.status;
  dom.status.disabled = allView;
}

function recordButton(record) {
  const selected = state.selectedKey === recordKey(record);
  const button = element("button", "record-item");
  button.type = "button";
  button.dataset.key = recordKey(record);
  button.setAttribute("aria-pressed", String(selected));

  const meta = element("div", "record-item__meta");
  append(
    meta,
    element("span", "record-item__id", record.id),
    element("span", "record-item__dataset", record.datasetMeta.short_label),
  );
  const footer = element("div", "record-item__footer");
  footer.append(statusPill(record.status));
  if (record.risk_flags.length) {
    footer.append(element("span", "risk-pill", `● ${record.risk_flags.length} 项提示`));
  }
  append(
    button,
    meta,
    safeTitleElement("h3", "", record.title),
    element("p", "record-item__subtitle", record.subtitle || "—"),
    footer,
  );
  button.addEventListener("click", () => selectRecord(record, true));
  return button;
}

function renderRecords() {
  const records = filteredRecords();
  const datasetLabel = state.dataset === "all" ? "全部类别" : getDataset(state.dataset).label;
  const preset = getQuickView(state.preset);
  dom.count.textContent = `${records.length} 条结果`;
  dom.context.textContent = preset
    ? `${datasetLabel} · 快捷任务：${preset.title}`
    : `${datasetLabel} · 只读快照`;
  dom.activePreset.hidden = !preset;
  if (preset) {
    const clear = element("button", "", "退出快捷任务");
    clear.type = "button";
    clear.addEventListener("click", () => {
      state.preset = null;
      renderQuickViews();
      renderRecords();
    });
    dom.activePreset.replaceChildren(
      document.createTextNode(`当前快捷任务：${preset.title} · ${records.length} 条`),
      clear,
    );
  } else {
    dom.activePreset.replaceChildren();
  }
  dom.clear.hidden = !(
    state.status !== "all" || state.riskOnly || state.query || state.preset || state.sort !== "id"
  );
  dom.exportCsv.disabled = records.length === 0;
  dom.copyViewLink.disabled = records.length === 0;

  if (!records.length) {
    const empty = element("div", "empty-results");
    append(
      empty,
      element("h3", "", "没有匹配记录"),
      element("p", "", "尝试缩短关键词，或清除状态与重点审阅筛选。"),
    );
    dom.list.replaceChildren(empty);
    state.selectedKey = null;
    clearHash();
    renderEmptyDetail("当前筛选没有可显示的记录");
    return;
  }

  if (!records.some((record) => recordKey(record) === state.selectedKey)) {
    state.selectedKey = recordKey(records[0]);
  }
  dom.list.replaceChildren(...records.map(recordButton));
  const selected = records.find((record) => recordKey(record) === state.selectedKey);
  if (selected) {
    updateHash(selected);
    renderDetail(selected);
  }
}

function renderEmptyDetail(message) {
  const wrapper = element("div", "detail-empty");
  append(
    wrapper,
    element("span", "", "↳"),
    element("h2", "", message),
    element("p", "", "页面不会改变源 JSONL；筛选仅影响当前显示。"),
  );
  dom.detail.replaceChildren(wrapper);
}

function showToast(message) {
  dom.toast.textContent = message;
  dom.toast.classList.add("is-visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => dom.toast.classList.remove("is-visible"), 1800);
}

async function copyRecordId(id) {
  try {
    await navigator.clipboard.writeText(id);
    showToast(`已复制 ${id}`);
  } catch (_error) {
    showToast(`记录 ID：${id}`);
  }
}

function downloadText(filename, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function downloadRecord(record) {
  downloadText(
    `${record.id}.json`,
    `${JSON.stringify(record.record, null, 2)}\n`,
    "application/json;charset=utf-8",
  );
}

function downloadFullSnapshot() {
  if (!state.payload) return;
  downloadText(
    `MitoEvidence-专家标注集-${state.payload.summary.total_records}条.json`,
    `${JSON.stringify(state.payload, null, 2)}\n`,
    "application/json;charset=utf-8",
  );
  showToast(`已导出 ${state.payload.summary.total_records} 条核验记录`);
}

function selectRecord(record, userInitiated = false) {
  state.selectedKey = recordKey(record);
  updateHash(record, userInitiated);
  dom.list.querySelectorAll(".record-item").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.key === state.selectedKey));
  });
  renderDetail(record);
  if (userInitiated && window.matchMedia("(max-width: 760px)").matches) {
    document.body.classList.add("mobile-detail-open");
    dom.detail.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function detailHeader(record) {
  const header = element("header", "detail-header");
  const top = element("div", "detail-header__top");
  const actions = element("div", "detail-actions");
  const source = element("a", "source-link", `仓库当前源文件 · L${record.source_line} ↗`);
  source.href = `${REPOSITORY_BLOB_URL}${record.source_path}#L${record.source_line}`;
  source.target = "_blank";
  source.rel = "noreferrer";
  const copy = element("button", "quiet-button", "复制 ID");
  copy.type = "button";
  copy.addEventListener("click", () => copyRecordId(record.id));
  const download = element("button", "quiet-button", "下载 JSON");
  download.type = "button";
  download.addEventListener("click", () => downloadRecord(record));
  const print = element("button", "quiet-button", "打印 / PDF");
  print.type = "button";
  print.addEventListener("click", () => window.print());
  append(actions, source, copy, download, print);
  append(top, element("span", "detail-kicker", `${record.datasetMeta.label} · ${record.id}`), actions);
  append(
    header,
    top,
    safeTitleElement("h2", "", record.title),
    element("p", "detail-subtitle", record.subtitle || record.datasetMeta.description),
    recordSummary(record),
  );
  if (record.risk_flags.length) {
    header.append(chipRow(record.risk_flags, "risk"));
  }
  return header;
}

function recordSummary(record) {
  const row = record.record;
  const summary = element("div", "record-summary");
  const facts = element("div", "record-summary__facts");
  let explanation = "这条记录用于定义评测与证据使用边界。";
  if (record.dataset === "pilot_questions") {
    append(
      facts,
      statusPill(row.answerability),
      chip(`${row.required_claims?.length || 0} 条必需主张`),
      chip(
        row.evidence_papers?.length && row.evidence_spans?.length
          ? "证据锚点已绑定"
          : "证据尚未闭环",
        row.evidence_papers?.length && row.evidence_spans?.length ? "default" : "risk",
      ),
    );
    explanation = "先按本题可回答性、必需主张和禁止推断确定答案边界，再回到原始论文补齐锚点。";
  } else if (record.dataset === "claim_reviews") {
    const usable = row.usable_for_beta_cell_evidence;
    append(
      facts,
      statusPill(row.ai_decision),
      confidencePill(row.ai_confidence),
      chip(
        usable === true ? "β 细胞证据：可用" : usable === false ? "β 细胞证据：不可用" : "β 细胞证据：未确定",
        usable === true ? "default" : usable === false ? "risk" : "neutral",
      ),
    );
    explanation = row.ai_reasoning || explanation;
  } else if (record.dataset === "terminology_rules") {
    append(facts, statusPill(row.detector), confidencePill(row.ai_confidence), chip(row.category));
    explanation = row.why || explanation;
  } else if (record.dataset === "review_pool") {
    append(
      facts,
      statusPill(row.fulltext?.status),
      chip(`${row.bibliography?.year || "年份未知"}`),
      chip(`${row.reference_count || 0} 条参考文献条目`, "neutral"),
    );
    explanation = row.recommended_uses?.[0] || row.pool_role || explanation;
  }
  append(summary, facts, paragraph(explanation));
  return summary;
}

function provenanceDetails(record) {
  const details = element("details", "provenance-details");
  details.append(element("summary", "", "查看标注身份、历史字段与文件哈希"));
  const strip = element("div", "provenance-strip");
  const provenance = [
    ["快照层指定", `项目负责人确认的单一汇总参考（${state.payload.manifest.designation}）`],
    ["源记录 annotator（历史字段）", record.record.annotator],
    ["源记录 review_status（历史字段）", record.record.review_status],
    ["本页源数据 SHA-256", record.datasetMeta.source_sha256],
  ];
  provenance.forEach(([label, value]) => {
    const cell = element("div", "provenance-cell");
    append(cell, element("span", "", label), element("strong", "", displayValue(value)));
    strip.append(cell);
  });
  details.append(strip);
  return details;
}

function navigateToRecord(record) {
  state.dataset = record.dataset;
  state.status = "all";
  state.riskOnly = false;
  state.preset = null;
  state.query = "";
  state.selectedKey = recordKey(record);
  updateHash(record, true);
  if (window.matchMedia("(max-width: 760px)").matches) {
    document.body.classList.add("mobile-detail-open");
  }
  dom.search.value = "";
  dom.risk.checked = false;
  renderQuickViews();
  renderTabs();
  renderStatusOptions();
  renderRecords();
  document.querySelector("#review")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function crossLinkButton(record, context) {
  const button = element("button", "cross-link");
  button.type = "button";
  button.setAttribute("aria-label", `打开 ${record.id} ${record.title}`);
  append(
    button,
    element("span", "cross-link__id", record.id),
    element("span", "cross-link__title", context || record.title),
    element("span", "cross-link__arrow", "→"),
  );
  button.addEventListener("click", () => navigateToRecord(record));
  return button;
}

function relatedRecords(record) {
  const records = allRecords();
  if (record.dataset === "pilot_questions") {
    const pmids = new Set(
      (record.record.source_reviews || [])
        .map((value) => String(value).match(/^PMID:(.+)$/)?.[1])
        .filter(Boolean),
    );
    return records
      .filter(
        (candidate) =>
          candidate.dataset === "review_pool" &&
          pmids.has(String(candidate.record.bibliography?.pmid || "")),
      )
      .map((candidate) => ({ record: candidate, context: `综述来源 · PMID ${candidate.record.bibliography.pmid}` }));
  }
  if (record.dataset === "claim_reviews") {
    return records
      .filter(
        (candidate) =>
          candidate.dataset === "terminology_rules" &&
          (candidate.record.observed_in_local_corpus || []).includes(record.record.statement_id),
      )
      .map((candidate) => ({ record: candidate, context: `命中该 Claim 的规则 · ${candidate.record.category}` }));
  }
  if (record.dataset === "terminology_rules") {
    const statementIds = new Set(record.record.observed_in_local_corpus || []);
    return records
      .filter(
        (candidate) =>
          candidate.dataset === "claim_reviews" && statementIds.has(candidate.record.statement_id),
      )
      .map((candidate) => ({ record: candidate, context: `本地实例 · ${candidate.record.triple}` }));
  }
  if (record.dataset === "review_pool") {
    const pmid = String(record.record.bibliography?.pmid || "");
    return records
      .filter(
        (candidate) =>
          candidate.dataset === "pilot_questions" &&
          (candidate.record.source_reviews || []).includes(`PMID:${pmid}`),
      )
      .map((candidate) => ({ record: candidate, context: `引用该综述的 Pilot · ${candidate.record.question_type}` }));
  }
  return [];
}

function relatedSection(record) {
  const related = relatedRecords(record);
  if (!related.length) return null;
  const links = element("div", "cross-links");
  related.forEach((item) => links.append(crossLinkButton(item.record, item.context)));
  return section("跨表关联", `${related.length} 条稳定 ID 对应`, links);
}

function renderPilot(record) {
  const row = record.record;
  const claims = element("div", "claim-grid");
  const allClaims = [
    ...(row.required_claims || []).map((claim) => ({ ...claim, requirement: "必需" })),
    ...(row.optional_claims || []).map((claim) => ({ ...claim, requirement: "可选" })),
  ];
  allClaims.forEach((claim, index) => {
    const card = element("article", "claim-card");
    const meta = element("div", "claim-card__meta");
    const badges = element("div", "claim-card__badges");
    append(
      badges,
      chip(claim.is_core ? "核心" : "非核心", claim.is_core ? "default" : "neutral"),
      confidencePill(claim.ai_confidence),
    );
    append(
      meta,
      element("span", "claim-card__number", `${claim.requirement} · ${String(index + 1).padStart(2, "0")}`),
      badges,
    );
    append(card, meta, paragraph(claim.text));
    claims.append(card);
  });

  const evidence = element("div", "evidence-stack");
  append(
    evidence,
    factGrid([
      ["综述来源", displayValue(row.source_reviews)],
      ["原始论文锚点", row.evidence_papers?.length ? displayValue(row.evidence_papers) : "空列表（尚未绑定）"],
      ["原文片段锚点", row.evidence_spans?.length ? displayValue(row.evidence_spans) : "空列表（尚未绑定）"],
    ]),
  );
  if (!row.evidence_papers?.length || !row.evidence_spans?.length) {
    evidence.append(notice("当前问题尚未绑定原始论文与原文片段；不能仅凭“可回答”状态宣称证据闭环完成。", "warning"));
  }

  return [
    section("问题边界", "问题级判定", factGrid([
      ["问题类型", row.question_type],
      ["可回答性", labelStatus(row.answerability)],
    ]), paragraph(row.scope)),
    section("应覆盖的主张", `${allClaims.length} 条`, claims),
    section("必填实验条件", `${row.required_context_slots?.length || 0} 项`, chipRow(row.required_context_slots)),
    section("证据绑定状态", "区分综述线索与原文锚点", evidence),
    section("已知冲突", "空列表不等于不存在冲突", listCard(row.known_conflicts, "当前记录未列出已知冲突")),
    section("禁止推断", "回答不得越过这些边界", listCard(row.prohibited_inferences)),
    section("保留待核事项", `${row.needs_human_verification?.length || 0} 项`, listCard(row.needs_human_verification)),
  ];
}

function mobileDetailNavigation(record) {
  const navigation = element("nav", "mobile-detail-nav");
  navigation.setAttribute("aria-label", "移动端记录导航");
  const records = filteredRecords();
  const index = records.findIndex((candidate) => recordKey(candidate) === recordKey(record));
  const back = element("button", "quiet-button", "← 返回列表");
  back.type = "button";
  back.addEventListener("click", () => {
    document.body.classList.remove("mobile-detail-open");
    document.querySelector("#records")?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  const paging = element("div", "mobile-detail-nav__paging");
  const previous = element("button", "quiet-button", "上一条");
  const next = element("button", "quiet-button", "下一条");
  previous.type = "button";
  next.type = "button";
  previous.disabled = index <= 0;
  next.disabled = index < 0 || index >= records.length - 1;
  previous.addEventListener("click", () => {
    if (index > 0) selectRecord(records[index - 1], false);
  });
  next.addEventListener("click", () => {
    if (index >= 0 && index < records.length - 1) selectRecord(records[index + 1], false);
  });
  append(paging, previous, element("span", "", `${index + 1}/${records.length}`), next);
  append(navigation, back, paging);
  return navigation;
}

function parseTriple(value) {
  const text = String(value || "");
  const match = text.match(/^(.*?)\s*--([^>]+?)-->\s*(.*?)$/);
  if (!match) return null;
  return { head: match[1].trim(), relation: match[2].trim(), tail: match[3].trim() };
}

function renderClaim(record) {
  const row = record.record;
  const parsed = parseTriple(row.triple);
  let triple;
  if (parsed) {
    triple = element("div", "triple-flow");
    append(
      triple,
      element("div", "triple-node", parsed.head),
      element("div", "triple-edge", parsed.relation),
      element("div", "triple-node", parsed.tail),
    );
  } else {
    triple = quote(row.triple);
  }
  return [
    section("候选关系", "仅表达被审记录，不自动等同于事实", triple),
    section("完整判定理由", "单一汇总结果", factGrid([
      ["结论", labelStatus(row.ai_decision)],
      ["置信度", labelStatus(row.ai_confidence)],
      ["可用于 β 细胞证据", row.usable_for_beta_cell_evidence],
      ["缺陷代码", displayValue(row.defect_codes)],
    ]), paragraph(row.ai_reasoning)),
    section("实验条件", `${Object.keys(row.recorded_conditions || {}).length} 个字段`, objectCards(row.recorded_conditions, "条件字段为空；跨模型复用存在风险")),
    section("原始证据片段", "引用仅供审阅，不代表已通过准入", quote(row.evidence_text)),
    section("来源定位", "论文与段落", factGrid([
      ["论文", row.paper_short],
      ["Statement ID", row.statement_id],
      ["Paper ID", row.paper_id],
      ["来源类型 / 章节", `${formatCode(row.source_type)} / ${formatCode(row.section)}`],
    ])),
    section("建议修改", "不自动写回源记录", objectCards(row.suggested_edits, "未提供修改建议")),
    section("保留待核事项", `${row.needs_human_verification?.length || 0} 项`, listCard(row.needs_human_verification)),
  ];
}

function renderTerm(record) {
  const row = record.record;
  const observed = row.observed_in_local_corpus;
  let observedContent;
  if (observed === null || observed === undefined) {
    observedContent = notice("未记录（null）：表示尚未检查本地语料，不等同于 0 个命中。", "warning");
  } else if (!observed.length) {
    observedContent = notice("已记录为空列表：当前没有登记本地命中。", "neutral");
  } else {
    observedContent = chipRow(observed);
  }
  return [
    section("推荐表述", "错误—正确成对呈现", compareCards("应避免", row.wrong, "推荐写法", row.correct)),
    section("为什么需要修正", "判定依据", paragraph(row.why)),
    section("示例对照", "用于理解，不替代原始证据", compareCards("错误示例", row.example_wrong, "正确示例", row.example_correct)),
    section("规则属性", "检测与评分映射", factGrid([
      ["类别", row.category],
      ["检测方式", labelStatus(row.detector)],
      ["关联评分维度", displayValue(row.maps_to_dimension)],
      ["置信度", labelStatus(row.ai_confidence)],
    ])),
    section("本地语料命中", observed === null ? "状态未知" : `${observed.length} 条`, observedContent),
    section("保留待核事项", `${row.needs_human_verification?.length || 0} 项`, listCard(row.needs_human_verification)),
  ];
}

function renderReview(record) {
  const row = record.record;
  const bibliography = row.bibliography || {};
  const fulltext = row.fulltext || {};
  const citation = factGrid([
    ["PMID", bibliography.pmid],
    ["PMCID", bibliography.pmcid],
    ["DOI", bibliography.doi],
    ["年份", bibliography.year],
  ]);
  if (bibliography.doi) {
    const doiLink = element("a", "doi-link", `打开 DOI：${bibliography.doi} ↗`);
    doiLink.href = `https://doi.org/${encodeURIComponent(bibliography.doi)}`;
    doiLink.target = "_blank";
    doiLink.rel = "noreferrer";
    citation.append(doiLink);
  }
  return [
    section("文献身份", "稳定标识与年份", citation),
    section("候选池定位", "综述用于导航，具体实验结论仍应回溯原始研究", factGrid([
      ["池内角色", formatCode(row.pool_role)],
      ["纳入判断", formatCode(row.ai_decision)],
      ["参考文献数", row.reference_count],
      ["Manifest 序号", row.source_manifest_index],
    ]), chipRow(row.coverage)),
    section("全文状态", labelStatus(fulltext.status), factGrid([
      ["状态", labelStatus(fulltext.status)],
      ["本地路径", fulltext.path],
      ["SHA-256", fulltext.sha256],
      ["错误记录", fulltext.error],
    ])),
    section("推荐用途", `${row.recommended_uses?.length || 0} 项`, listCard(row.recommended_uses)),
    section("使用限制", "生成回答时必须保留", listCard(row.caveats, "当前记录未列出额外限制")),
    section("保留待核事项", `${row.required_human_checks?.length || 0} 项`, listCard(row.required_human_checks)),
  ];
}

function rawRecord(record) {
  const details = element("details", "raw-details");
  details.append(element("summary", "", "查看原始 JSON 记录"));
  details.append(element("pre", "", JSON.stringify(record.record, null, 2)));
  return details;
}

function renderDetail(record) {
  const bodyByDataset = {
    pilot_questions: renderPilot,
    claim_reviews: renderClaim,
    terminology_rules: renderTerm,
    review_pool: renderReview,
  };
  const renderer = bodyByDataset[record.dataset];
  if (!renderer) {
    renderEmptyDetail("此数据类别尚无展示模板");
    return;
  }
  const nodes = [
    mobileDetailNavigation(record),
    detailHeader(record),
    ...renderer(record),
    relatedSection(record),
    provenanceDetails(record),
    rawRecord(record),
  ].filter(Boolean);
  dom.detail.replaceChildren(...nodes);
}

function applyHashSelection() {
  const target = parseHash();
  if (!target) return false;
  const match = allRecords().find(
    (record) => record.dataset === target.dataset && record.id === target.id,
  );
  if (!match) return false;
  state.dataset = target.dataset;
  state.status = "all";
  state.query = "";
  state.riskOnly = false;
  state.preset = null;
  state.sort = "id";
  state.selectedKey = recordKey(match);
  return true;
}

function csvCell(value) {
  let text = value === null || value === undefined
    ? ""
    : typeof value === "object"
      ? JSON.stringify(value)
      : String(value);
  if (/^[=+\-@]/.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
}

function exportFilteredCsv() {
  const records = filteredRecords();
  const headers = [
    "dataset",
    "id",
    "status",
    "confidence",
    "title",
    "risk_flags",
    "source_path",
    "source_line",
    "record_json",
  ];
  const lines = [headers.map(csvCell).join(",")];
  records.forEach((record) => {
    lines.push(
      [
        record.dataset,
        record.id,
        record.status,
        record.confidence,
        record.title,
        record.risk_flags.join(" | "),
        record.source_path,
        record.source_line,
        record.record,
      ].map(csvCell).join(","),
    );
  });
  downloadText(
    `MitoEvidence-标注筛选-${records.length}条.csv`,
    `\ufeff${lines.join("\r\n")}\r\n`,
    "text/csv;charset=utf-8",
  );
  showToast(`已导出 ${records.length} 条记录`);
}

async function copyCurrentLink() {
  try {
    await navigator.clipboard.writeText(window.location.href);
    showToast("已复制当前记录链接");
  } catch (_error) {
    showToast("浏览器未授权复制，请从地址栏复制");
  }
}

function openProvenanceDialog() {
  if (typeof dom.provenanceDialog.showModal === "function") {
    dom.provenanceDialog.showModal();
  } else {
    dom.provenanceDialog.setAttribute("open", "");
  }
}

function bindControls() {
  dom.search.addEventListener("input", (event) => {
    if (!state.payload) return;
    state.query = event.target.value;
    renderRecords();
  });
  dom.status.addEventListener("change", (event) => {
    if (!state.payload) return;
    state.status = event.target.value;
    state.preset = null;
    renderQuickViews();
    renderRecords();
  });
  dom.risk.addEventListener("change", (event) => {
    if (!state.payload) return;
    state.riskOnly = event.target.checked;
    renderRecords();
  });
  dom.sort.addEventListener("change", (event) => {
    if (!state.payload) return;
    state.sort = event.target.value;
    renderRecords();
  });
  dom.clear.addEventListener("click", () => {
    if (!state.payload) return;
    state.status = "all";
    state.riskOnly = false;
    state.preset = null;
    state.sort = "id";
    state.query = "";
    dom.search.value = "";
    dom.risk.checked = false;
    dom.sort.value = "id";
    renderQuickViews();
    renderTabs();
    renderStatusOptions();
    renderRecords();
  });
  dom.provenanceButton.addEventListener("click", openProvenanceDialog);
  dom.scopeNote.addEventListener("click", openProvenanceDialog);
  dom.downloadJson.addEventListener("click", downloadFullSnapshot);
  dom.exportCsv.addEventListener("click", exportFilteredCsv);
  dom.copyViewLink.addEventListener("click", copyCurrentLink);
  document.addEventListener("keydown", (event) => {
    const tag = document.activeElement?.tagName?.toLowerCase();
    const editable = ["input", "textarea", "select"].includes(tag) || document.activeElement?.isContentEditable;
    if (event.key === "/" && !editable) {
      event.preventDefault();
      dom.search.focus();
    }
    if (event.key === "Escape" && document.activeElement === dom.search && dom.search.value) {
      dom.search.value = "";
      state.query = "";
      renderRecords();
    }
  });
  const restoreFromLocation = () => {
    if (
      window.location.protocol === "file:" &&
      fileHashWrittenByApp !== null &&
      window.location.hash === fileHashWrittenByApp
    ) {
      fileHashWrittenByApp = null;
      return;
    }
    if (!state.payload || !applyHashSelection()) return;
    if (window.matchMedia("(max-width: 760px)").matches) {
      document.body.classList.add("mobile-detail-open");
    }
    dom.search.value = "";
    dom.risk.checked = false;
    dom.sort.value = "id";
    renderQuickViews();
    renderTabs();
    renderStatusOptions();
    renderRecords();
  };
  window.addEventListener("hashchange", restoreFromLocation);
  window.addEventListener("popstate", () => {
    if (window.location.protocol !== "file:") restoreFromLocation();
  });
}

function setControlsDisabled(disabled) {
  dom.search.disabled = disabled;
  dom.risk.disabled = disabled;
  dom.clear.disabled = disabled;
  dom.sort.disabled = disabled;
  dom.downloadJson.disabled = disabled;
  dom.exportCsv.disabled = disabled;
  dom.copyViewLink.disabled = disabled;
  if (disabled) dom.status.disabled = true;
}

function renderLoadError(error) {
  const wrapper = element("div", "error-state");
  append(
    wrapper,
    element("strong", "", "无法载入审阅数据"),
    element("p", "", error instanceof Error ? error.message : String(error)),
  );
  dom.list.replaceChildren(wrapper);
  dom.count.textContent = "载入失败";
  dom.context.textContent = "请检查生成文件与网络连接";
  renderEmptyDetail("数据尚未载入");
}

async function start() {
  setControlsDisabled(true);
  bindControls();
  try {
    if (window.__MITOEVIDENCE_ANNOTATIONS__) {
      state.payload = window.__MITOEVIDENCE_ANNOTATIONS__;
    } else {
      const response = await fetch(DATA_URL, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}：${response.statusText}`);
      state.payload = await response.json();
    }
    if (state.payload.schema_version !== "mitoevidence.annotation-review-site.v2") {
      throw new Error(`不支持的数据版本：${state.payload.schema_version || "missing"}`);
    }
    dom.date.textContent = formatDate(state.payload.manifest.confirmed_at);
    dom.hash.textContent = `manifest ${state.payload.manifest_sha256.slice(0, 12)}…`;
    setControlsDisabled(false);
    const restoredFromHash = applyHashSelection();
    if (restoredFromHash && window.matchMedia("(max-width: 760px)").matches) {
      document.body.classList.add("mobile-detail-open");
    }
    renderFlow();
    renderMetrics();
    renderCoverage();
    renderDistributions();
    renderQuickViews();
    renderTabs();
    renderStatusOptions();
    renderRecords();
  } catch (error) {
    renderLoadError(error);
  }
}

start();
