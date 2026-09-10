"""Build a portable, source-held-out CPU tool environment; never train a model.

Only named train/development files are opened. Original labels, XML, database
and frozen test files are never modified. Output is a new bounded tool task
set with programmatic extractive/arithmetical targets, not expert gold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from collections import Counter, defaultdict

VERSION = "rp-local-snapshot.v1"
SECRET = re.compile(r"sk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._-]{16,}|(?:api[_ -]?key|access[_ -]?token|password)\s*[:=]\s*\S+", re.I)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _assert_text(value):
    if not isinstance(value, str) or SECRET.search(value):
        raise ValueError("non-text or possible credential in actor-visible material")


def build(primary_root, hy_root, output):
    primary_root, hy_root, output = Path(primary_root).resolve(), Path(hy_root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty; snapshot versions are immutable")
    manifest = json.loads((primary_root / "manifest.json").read_text())
    source_rows, input_files = [], {}
    for filename, split in (("train.jsonl", "train"), ("development.jsonl", "dev")):
        path = primary_root / filename
        digest = file_hash(path)
        if manifest["files"][filename]["sha256"] != digest:
            raise ValueError("original task file hash mismatch")
        input_files[f"primary_tool_tasks_v3/{filename}"] = digest
        for row in records(path):
            if row["split"] != ("development" if split == "dev" else split):
                raise ValueError("source split mismatch")
            source_rows.append((split, row))
    passages, views, tasks, scorers, provenance = {}, {}, [], [], []
    paper_splits, source_splits, pair_splits, pair_sides = {}, {}, {}, defaultdict(set)
    checked_xml = {}
    for split, row in source_rows:
        if row["provenance"].get("label_origin") != "programmatic_exact_source_v1":
            raise ValueError("source labels are not the expected programmatic extraction task")
        _assert_text(row["question"])
        _assert_text(row["retrieval_query"])
        paper_ids = row["provenance"]["paper_ids"]
        for paper in paper_ids:
            if paper in paper_splits and paper_splits[paper] != split:
                raise ValueError("paper leaked between train and dev")
            paper_splits[paper] = split
        pair = "pair-" + canonical_hash(row["pair_id"])[:24]
        if pair in pair_splits and pair_splits[pair] != split:
            raise ValueError("paired tasks crossed splits")
        pair_splits[pair] = split
        side = "available" if row["expected"]["answerability"] == "answerable" else "view_without_target"
        pair_sides[pair].add(side)
        authorized = []
        for item in row["corpus_records"]:
            _assert_text(item["text"])
            source_sha = item["source_sha256"]
            if source_sha in source_splits and source_splits[source_sha] != split:
                raise ValueError("same raw source hash crossed train/dev")
            source_splits[source_sha] = split
            source_path = (hy_root / item["source_path"]).resolve()
            if not source_path.is_relative_to(hy_root / "data/training/local/epmc_primary_sources_v1"):
                raise ValueError("source XML is outside the approved primary corpus")
            if str(source_path) not in checked_xml:
                checked_xml[str(source_path)] = file_hash(source_path)
            if checked_xml[str(source_path)] != source_sha:
                raise ValueError("source XML hash mismatch")
            evidence_id = item["evidence_id"]
            record = {"evidence_id": evidence_id, "paper_id": item["paper_id"], "text": item["text"],
                      "section": item.get("section", ""), "source_sha256": source_sha,
                      "text_sha256": hashlib.sha256(item["text"].encode()).hexdigest(),
                      "locator": {"kind": "source_paragraph", "paragraph_id": evidence_id},
                      "source_kind": "primary_xml_extracted_paragraph", "split": split,
                      "title": row["provenance"]["source_identifiers"].get("title", ""),
                      "doi": row["provenance"]["source_identifiers"].get("doi"),
                      "license": row["provenance"].get("license"), "label_scope": "extractive_not_scientific_adjudication"}
            if evidence_id in passages and passages[evidence_id] != record:
                raise ValueError("same evidence id has inconsistent source bytes or metadata")
            passages[evidence_id] = record
            authorized.append(evidence_id)
        view = {"passage_ids": sorted(authorized), "measurement_row_ids": [],
                "scope": "authorized_snapshot_only", "paper_ids": sorted(paper_ids)}
        view_id = "view-" + canonical_hash(view)[:24]
        views[view_id] = {"id": view_id, **view}
        task_id = "rp-" + canonical_hash(row["task_id"])[:24]
        actor = {"question": row["question"], "retrieval_query": row["retrieval_query"],
                 "paper_ids": sorted(paper_ids), "scope": "authorized_snapshot_only",
                 "task_contract": "Read exact source material before making an extractive answer. An unavailable authorized view is not proof of absence from science or the experiment."}
        task = {"id": task_id, "split": split, "group_id": row["question_family_id"], "pair_id": pair,
                "side": side, "variant": row["task_type"], "question": row["question"],
                "actor_observation": actor, "view_id": view_id, "view_sha256": canonical_hash(view),
                "source_version": VERSION, "task_kind": "text_evidence"}
        expected = row["expected"]
        allowed = expected.get("allowed_evidence_ids", [])
        if side == "available" and (not allowed or not set(allowed).issubset(authorized)):
            raise ValueError("answerable target is not actually readable")
        if side != "available" and any(row["retrieval_query"].casefold() in passages[e]["text"].casefold() for e in authorized):
            raise ValueError("insufficient view still contains matching target")
        scorer = {"task_id": task_id, "kind": "exact_passage", "answerability": expected["answerability"],
                  "allowed_evidence_ids": allowed, "expected_claims": expected.get("claims", []),
                  "anchor": row["retrieval_query"], "scope": "authorized_snapshot_only", "expert_gold": False}
        tasks.append(task)
        scorers.append(scorer)
        provenance.append({"task_id": task_id, "source_task_id": row["task_id"], "source_record_sha256": canonical_hash(row),
                           "source_corpus_snapshot_sha256": row["corpus_snapshot_sha256"], "source_file": f"primary_tool_tasks_v3/{'train' if split == 'train' else 'development'}.jsonl"})
    if any(sides != {"available", "view_without_target"} for sides in pair_sides.values()):
        raise ValueError("counterfactual pair is incomplete")

    # One source-isolated repair pair per paper. Reset executes this malformed
    # query through the real tool validator and shows the resulting feedback;
    # the target policy repairs it, not a pre-written success observation.
    repair_pairs = {}
    for task in tasks:
        repair_pairs.setdefault((task["split"], task["group_id"]), task["pair_id"])
    for task in list(tasks):
        if repair_pairs[(task["split"], task["group_id"])] != task["pair_id"]:
            continue
        original_scorer = next(row for row in scorers if row["task_id"] == task["id"])
        repaired = {**task, "id": "rp-" + canonical_hash([task["id"], "repair"])[:24],
                    "pair_id": "pair-" + canonical_hash([task["pair_id"], "repair"])[:24],
                    "variant": "tool_error_repair", "setup_action": {"tool": "search_evidence", "arguments": {"query": ""}}}
        tasks.append(repaired)
        scorers.append({**original_scorer, "task_id": repaired["id"]})
        provenance.append({"task_id": repaired["id"], "derived_from": task["id"], "variant": "real_invalid_query_feedback", "human_review": False})

    measurement_rows = _measurements(hy_root, input_files)
    grouped = defaultdict(list)
    for row in measurement_rows:
        grouped[(row["split"], row["experiment_id"], row["wavelength_nm"])].append(row)
    for (split, experiment, wavelength), rows in grouped.items():
        for aggregation in ("mean", "median"):
            request = {"experiment_id": experiment, "metric": "median_all_planes", "aggregation": aggregation,
                       "wavelength_nm": int(wavelength), "unit": "a.u."}
            question = f"在当前授权的只读测量快照中，按 islet 先聚合重复图像，再计算 {experiment} {wavelength} nm 通道 median_all_planes 的{aggregation}。返回工具结果ID、数值及单位；不要解释为 ATP、膜电位或小鼠级推断。若视图没有这些记录，只说明授权范围不足。"
            pair = "pair-" + canonical_hash(request)[:24]
            for side, ids in (("available", [r["row_id"] for r in rows]), ("view_without_target", [])):
                view = {"passage_ids": [], "measurement_row_ids": ids, "scope": "authorized_snapshot_only", "paper_ids": []}
                view_id = "view-" + canonical_hash(view)[:24]
                views[view_id] = {"id": view_id, **view}
                task_id = "rp-" + canonical_hash([request, side])[:24]
                actor = {"question": question, "scope": "authorized_snapshot_only", "measurement_request": request,
                         "task_contract": "Finite whole-FOV intensity description only. Snapshot omission does not establish that the original experiment never measured a variable."}
                tasks.append({"id": task_id, "split": split, "group_id": "experiment:" + experiment, "pair_id": pair,
                              "side": side, "variant": "metric_" + aggregation, "question": question, "actor_observation": actor,
                              "view_id": view_id, "view_sha256": canonical_hash(view), "source_version": VERSION, "task_kind": "finite_metric"})
                scorers.append({"task_id": task_id, "kind": "finite_metric", "answerability": "answerable" if ids else "insufficient",
                                "measurement_request": request, "scope": "authorized_snapshot_only", "expert_gold": False})
    groups = defaultdict(set)
    for task in tasks:
        groups[task["group_id"]].add(task["split"])
    if any(len(parts) != 1 for parts in groups.values()):
        raise ValueError("group leakage")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "passages.jsonl", sorted(passages.values(), key=lambda r: r["evidence_id"]))
    write_jsonl(output / "views.jsonl", sorted(views.values(), key=lambda r: r["id"]))
    write_jsonl(output / "tasks.jsonl", sorted(tasks, key=lambda r: r["id"]))
    write_jsonl(output / "scorers.jsonl", sorted(scorers, key=lambda r: r["task_id"]))
    write_jsonl(output / "provenance.jsonl", provenance)
    fields = list(measurement_rows[0]) if measurement_rows else ["row_id"]
    with (output / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(measurement_rows)
    result = {"schema_version": VERSION, "counts": {"tasks": len(tasks), "passages": len(passages), "measurement_rows": len(measurement_rows),
               "splits": dict(Counter(t["split"] for t in tasks)), "task_kinds": dict(Counter(t["task_kind"] for t in tasks))},
              "input_sha256": input_files, "files": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
              "source_paper_splits": paper_splits, "source_hash_splits": source_splits,
              "limitations": ["No frozen-test tasks or labels were read into this build.",
                 "Text targets are exact-source retrieval, not expert scientific adjudication.",
                 "Actor receives only actor_observation, not id/split/pair/side/scorer or global stores.",
                 "Two authorized views differ by target availability; absence is scoped to this environment.",
                 "Measurements are copied from an existing read-only compendium; image weights remain fixed and no image analysis was rerun.",
                 "Whole-FOV intensity includes background; islet aggregation is descriptive, not independent mouse replication.",
                 "This is an offline bounded environment, not all production APIs or a general causal reasoning benchmark."]}
    (output / "MANIFEST.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _measurements(hy_root, inputs):
    """Explicit train/dev experiments only; no arbitrary SQL interface is exposed."""
    import duckdb  # Build-time dependency only; portable environment uses csv/std-lib.
    split_file = hy_root / "training_data/splits/family_a_splits.json"
    splits = json.loads(split_file.read_text())
    inputs["family_a_splits.json"] = file_hash(split_file)
    db = hy_root / "data/compendium/v0.1.0-20260907/isletmito.duckdb"
    inputs["compendium/isletmito.duckdb"] = file_hash(db)
    selected = {"train": "confocal_20250507_ctrl_tmrm_hoechst", "dev": "confocal_20250704_6p5w_wt_ob_pkmdr"}
    output = []
    with duckdb.connect(str(db), read_only=True) as connection:
        for split, experiment in selected.items():
            if experiment not in splits[split] or any(experiment in splits[s] for s in {"train", "dev", "test"} - {split}):
                raise ValueError("measurement experiment split is not exclusive")
            query = "SELECT experiment_id, condition_id, dish_id, islet_id, image_id, wavelength_nm, median_all_planes, dataset_version, pipeline_version, analysis_unit_recommendation FROM l4_islet_intensity_summary WHERE experiment_id = ? AND role = 'mito' ORDER BY image_id"
            cursor = connection.execute(query, [experiment])
            names = [d[0] for d in cursor.description]
            for values in cursor.fetchall():
                original = dict(zip(names, values))
                value = original["median_all_planes"]
                if not original["islet_id"] or original["wavelength_nm"] is None or value is None or not math.isfinite(float(value)):
                    continue
                row = {"row_id": "measure-" + canonical_hash(original)[:24], "split": split,
                       **{key: original[key] for key in ("experiment_id", "condition_id", "dish_id", "islet_id", "image_id", "wavelength_nm")},
                       "metric": "median_all_planes", "value": float(value), "unit": "a.u.",
                       "method": "whole-FOV median intensity including background; no segmentation-derived measurement",
                       "dataset_version": original["dataset_version"], "pipeline_version": original["pipeline_version"],
                       "source_row_sha256": canonical_hash(original), "source_table": "l4_islet_intensity_summary",
                       "source_database_sha256": inputs["compendium/isletmito.duckdb"],
                       "limitation": original["analysis_unit_recommendation"]}
                output.append(row)
    if not output:
        raise ValueError("no real authorized measurements available")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", required=True, type=Path)
    parser.add_argument("--hy-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.primary_root, args.hy_root, args.output), ensure_ascii=False, indent=2))
