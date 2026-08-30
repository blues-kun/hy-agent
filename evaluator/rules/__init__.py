"""确定性规则层。

方案 7.3：规则层处理 DOI/PMID、元数据、数字、单位、重复引用、格式和硬性安全条件；
语义判定（主张与原文的支持关系）不在本层，由 Judge 层负责。

  - identifier_check  D1 引用真实性：标识符规范化 + Crossref/NCBI 批量核验；
  - structure_check   D7 流程必需项与 D9 清单的可配置检查器；
  - numeric_check     D8 数字与单位的抽取与比对。
  - terminology_check D8 本地版本化术语三态初筛（不冒充完整 MeSH/GO 真值）。
"""
