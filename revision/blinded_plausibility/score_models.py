#!/usr/bin/env python3
"""
score_models.py
===============
Apply the SAME pre-registered, blinded plausibility rule encoded in
``score_blinded.py`` to additional models (RefH, BoxE) on the SAME 297 forward
clinical-indication queries, reusing the SAME on-disk Open Targets cache and the
SAME model-independent popularity prior.

The 297 forward queries are identical across models (same KG, same test split,
same query order), so the per-query target gene(s) and their Open Targets
associated-disease lists are model-independent: only the model's TOP-50
predictions differ. We therefore run fully OFFLINE against the cache built by the
original RotE run -- no network, CPU only.

Usage:
    python3 score_models.py --trial_dir <dir_with_interpret_topk_50.json> --label RefH
Writes <here>/summary_<label>.json and appends a row to model_comparison.csv via
the companion build step.
"""
import os
import sys
import csv
import json
import random
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import score_blinded as sb   # noqa: E402  (reuse the pre-registered logic)


def score_model(trial_dir, label):
    interpret_json = os.path.join(trial_dir, "interpret_topk_50.json")
    interp = sb.load_json(interpret_json)
    if not interp or "results" not in interp:
        raise SystemExit(f"missing/invalid interpret json: {interpret_json}")
    results = interp["results"]

    # SHARED, model-independent popularity prior (training-set indication freqs)
    prior = sb.load_json(sb.PRIOR_JSON)
    prior_counts = prior["counts"]
    prior_names = list(prior_counts.keys())
    prior_weights = [prior_counts[k] for k in prior_names]

    # SHARED caches (OFFLINE: do not hit the network)
    hgnc_cache = sb.load_json(sb.HGNC_CACHE, {})
    if not hgnc_cache and os.path.exists(sb.REF_HGNC_CACHE):
        hgnc_cache = dict(sb.load_json(sb.REF_HGNC_CACHE, {}))
    ot_cache = sb.load_json(sb.OT_CACHE, {})

    head_index, tail_index = sb.load_kg()
    live = False  # OFFLINE

    # resolve target gene symbols for every query (model-independent)
    for q in results:
        sp = q["head"]
        _, mab = sb.resolve_mab(sp, head_index)
        hgncs = sb.resolve_targets(mab, head_index, tail_index)
        symbols = []
        for h in hgncs:
            s = sb.hgnc_to_symbol(h, hgnc_cache, live)
            if s:
                symbols.append(s)
        q["_mab"] = mab
        q["_symbols"] = symbols

    rng = random.Random(sb.RANDOM_SEED)
    baseline_cache = {}
    rows = []
    coverage_specific_seen = 0
    coverage_matched = 0

    for q in results:
        symbols = q.get("_symbols", [])
        ot_union, ok = [], True
        for s in symbols:
            a = ot_cache.get(s)
            if a is None:
                ok = False
            else:
                ot_union.extend(a)
        top_ents = [item["ent"] for item in q["top_k"][:sb.TOP_K]]

        if not symbols or not ok or not ot_union:
            rows.append({"query_idx": q["query_idx"], "mab": q.get("_mab"),
                         "study_product": q["head"], "genes": ";".join(symbols),
                         "n_specific": "", "n_plausible": "", "model_fraction": "",
                         "prior_fraction": "", "enrichment": "", "n_unmatched": "",
                         "status": "no_target_or_ot"})
            continue

        sc = sb.score_query_topk(top_ents, ot_union, sb.PLAUSIBILITY_THRESHOLD)

        n_unmatched = 0
        for t in sc["tags"]:
            if t["tag"] == "GENERIC":
                continue
            coverage_specific_seen += 1
            if t["tag"] == "PLAUSIBLE":
                coverage_matched += 1
            else:
                n_unmatched += 1

        key = tuple(sorted(symbols))
        if key not in baseline_cache:
            baseline_cache[key] = [
                sb.classify_pred(nm, ot_union, sb.PLAUSIBILITY_THRESHOLD)[0]
                for nm in prior_names]
        tag_by_idx = baseline_cache[key]
        prior_frac = sb.baseline_fraction(prior_names, prior_weights, ot_union,
                                          sb.PLAUSIBILITY_THRESHOLD, rng, tag_by_idx)

        model_frac = sc["fraction"]
        enrichment = (model_frac / prior_frac) if (model_frac is not None
                                                   and prior_frac and prior_frac > 0) else None
        rows.append({"query_idx": q["query_idx"], "mab": q.get("_mab"),
                     "study_product": q["head"], "genes": ";".join(symbols),
                     "n_specific": sc["n_specific"], "n_plausible": sc["n_plausible"],
                     "model_fraction": round(model_frac, 4) if model_frac is not None else "",
                     "prior_fraction": round(prior_frac, 4),
                     "enrichment": round(enrichment, 4) if enrichment is not None else "",
                     "n_unmatched": n_unmatched, "status": "scored"})

    # write per-query CSV
    out_csv = os.path.join(HERE, f"results_297_{label}.csv")
    fieldnames = ["query_idx", "mab", "study_product", "genes", "n_specific",
                  "n_plausible", "model_fraction", "prior_fraction", "enrichment",
                  "n_unmatched", "status"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    scored = [r for r in rows if r["status"] == "scored"]
    model_fracs = [r["model_fraction"] for r in scored if r["model_fraction"] != ""]
    prior_fracs = [r["prior_fraction"] for r in scored if r["prior_fraction"] != ""]
    enr = [r["enrichment"] for r in scored if r["enrichment"] != ""]
    ci = sb.bootstrap_ci(enr, lambda s: sum(s) / len(s) if s else 0.0,
                         sb.N_BOOTSTRAP, sb.RANDOM_SEED)

    summary = {
        "label": label,
        "trial_dir": trial_dir,
        "model_mrr": interp.get("mrr"),
        "n_queries_total": len(results),
        "n_queries_scored": len(scored),
        "n_queries_unscorable": len(results) - len(scored),
        "model_fraction_mean": sb.mean(model_fracs),
        "model_fraction_median": sb.median(model_fracs),
        "baseline_fraction_mean": sb.mean(prior_fracs),
        "baseline_fraction_median": sb.median(prior_fracs),
        "enrichment_mean": sb.mean(enr),
        "enrichment_median": sb.median(enr),
        "enrichment_bootstrap_ci95": {"low": ci[0], "high": ci[1]},
        "frac_queries_enrichment_gt1": (sum(1 for e in enr if e > 1) / len(enr)) if enr else None,
        "normalisation_coverage": {
            "specific_predictions_scored": coverage_specific_seen,
            "specific_predictions_matched_OT": coverage_matched,
            "coverage_fraction": (coverage_matched / coverage_specific_seen
                                  if coverage_specific_seen else None),
        },
        "pre_registration": {
            "threshold": sb.PLAUSIBILITY_THRESHOLD, "top_k": sb.TOP_K,
            "n_prior_draws": sb.N_PRIOR_DRAWS, "prior_draw_size": sb.PRIOR_DRAW_SIZE,
            "random_seed": sb.RANDOM_SEED, "ot_diseases_per_gene": sb.OT_DISEASES_PER_GENE,
            "blinded": True, "offline_cache": True,
        },
    }
    out_summary = os.path.join(HERE, f"summary_{label}.json")
    sb.save_json(out_summary, summary)
    print(f"[{label}] scored {len(scored)}/{len(results)}  "
          f"model_frac={summary['model_fraction_mean']:.4f}  "
          f"baseline={summary['baseline_fraction_mean']:.4f}  "
          f"enrichment={summary['enrichment_mean']:.4f} "
          f"CI=[{ci[0]:.4f},{ci[1]:.4f}]  "
          f"frac>1={summary['frac_queries_enrichment_gt1']:.4f}")
    print(f"  wrote {out_summary}\n  wrote {out_csv}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial_dir", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    score_model(a.trial_dir, a.label)
