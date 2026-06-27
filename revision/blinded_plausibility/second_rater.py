#!/usr/bin/env python3
"""
second_rater.py
===============
Independent AUTOMATED second rater for the blinded plausibility tags, and
inter-rater agreement (Cohen's kappa + percent agreement) against the primary
rater, on the RotE TOP-20 predictions across the 297 forward queries.

This is the reproducible substitute for a reviewer's "second human rater": a
second automated rater that scores the SAME predictions on a DIFFERENT Open
Targets evidence axis. It is NOT a human.

Raters (all at the SAME matched stringency, score >= 0.01, on the SAME OT
snapshot fetched here so that disagreement reflects the EVIDENCE AXIS, not a
version drift):
  * RATER A (primary axis): Open Targets OVERALL association score
        -- the composite, harmonic-sum aggregation across all datatypes;
           this is the rule used in the headline analysis.
  * RATER B (independent axis, HEADLINE): Open Targets 'clinical' DATATYPE score
        -- clinical-precedence / known-drug evidence (ChEMBL drug-indication
           records that a drug against this target is in trials/approved for the
           disease). This is the OT "known_drug" evidence axis (its datatype id
           is 'clinical' in the v26 schema) and is the clinically-relevant axis
           distinct from the composite OVERALL score.
  * RATER C (additional independent axis): 'genetic_association' DATATYPE score
        -- human-genetics evidence (GWAS / ClinVar / gene-burden), an orthogonal
           but sparse axis (germline genetics is rare for somatic cancers), used
           as a stricter sensitivity comparator.

A prediction is tagged on the SAME normalisation + token-overlap matching rule as
the primary scorer (``score_blinded.classify_pred``); only the per-disease score
field changes between raters. GENERIC bucket predictions are excluded from BOTH
raters (identically), so they never enter the agreement computation.

CPU only, no torch. Network is used ONCE to fetch datatype scores (cached).
"""
import os
import sys
import csv
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_blinded as sb  # noqa: E402

DATATYPE_CACHE = os.path.join(sb.CACHE_DIR, "gene_ot_datatype.json")
THRESHOLD = sb.PLAUSIBILITY_THRESHOLD  # 0.01, matched stringency
TOP_K = sb.TOP_K                       # 20

FETCH_QUERY = """
query A($ens: String!, $size: Int!) {
  target(ensemblId: $ens) {
    approvedSymbol
    associatedDiseases(page: {index:0, size:$size}) {
      rows {
        score
        datatypeScores { id score }
        disease { name }
      }
    }
  }
}"""


def fetch_datatype(symbols, ensembl_cache, live):
    cache = sb.load_json(DATATYPE_CACHE, {})
    need = [s for s in symbols if s not in cache]
    print(f"datatype cache: {len(symbols)} genes, {len(need)} to fetch")
    for i, sym in enumerate(need):
        ens = ensembl_cache.get(sym)
        if not ens:
            cache[sym] = None
            continue
        if not live:
            continue
        res = sb.http_post_json(sb.OT_GRAPHQL,
                                {"query": FETCH_QUERY,
                                 "variables": {"ens": ens, "size": sb.OT_DISEASES_PER_GENE}})
        tgt = (res.get("data", {}) or {}).get("target") or {}
        rows = (tgt.get("associatedDiseases") or {}).get("rows", []) or []
        out = []
        for row in rows:
            name = ((row.get("disease") or {}).get("name") or "").strip().lower()
            if not name:
                continue
            overall = row.get("score")
            dts = {d["id"]: round(float(d["score"]), 4)
                   for d in (row.get("datatypeScores") or [])}
            out.append([name, round(float(overall), 4) if overall is not None else 0.0, dts])
        cache[sym] = out
        sb.save_json(DATATYPE_CACHE, cache)
        print(f"  [{i+1}/{len(need)}] {sym} ens={ens} diseases={len(out)}")
        time.sleep(sb.API_SLEEP)
    sb.save_json(DATATYPE_CACHE, cache)
    return cache


def view(dt_rows, axis):
    """Build a [[name, score_on_axis], ...] list for a given evidence axis."""
    if dt_rows is None:
        return None
    out = []
    for name, overall, dts in dt_rows:
        if axis == "overall":
            out.append([name, overall])
        else:
            out.append([name, dts.get(axis, 0.0)])
    return out


def cohen_kappa(pairs):
    """pairs: list of (tagA, tagB) over {'PLAUSIBLE','OFF-TARGET'}."""
    n = len(pairs)
    if n == 0:
        return None, None
    cats = ["PLAUSIBLE", "OFF-TARGET"]
    obs = sum(1 for a, b in pairs if a == b) / n
    pa = {c: sum(1 for a, _ in pairs if a == c) / n for c in cats}
    pb = {c: sum(1 for _, b in pairs if b == c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    kappa = (obs - pe) / (1 - pe) if (1 - pe) != 0 else None
    return obs, kappa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    live = not args.offline

    interp = sb.load_json(os.path.join(sb.TRIAL_DIR, "interpret_topk_50.json"))
    results = interp["results"]
    hgnc_cache = sb.load_json(sb.HGNC_CACHE, {})
    if not hgnc_cache and os.path.exists(sb.REF_HGNC_CACHE):
        hgnc_cache = dict(sb.load_json(sb.REF_HGNC_CACHE, {}))
    ensembl_cache = sb.load_json(sb.ENSEMBL_CACHE, {})
    hi, ti = sb.load_kg()

    # resolve symbols per query and collect the unique set
    symset = set()
    for q in results:
        _, mab = sb.resolve_mab(q["head"], hi)
        syms = [sb.hgnc_to_symbol(h, hgnc_cache, False)
                for h in sb.resolve_targets(mab, hi, ti)]
        q["_symbols"] = [s for s in syms if s]
        symset.update(q["_symbols"])

    dt_cache = fetch_datatype(sorted(symset), ensembl_cache, live)

    # snapshot version
    api_ver = None
    if live:
        try:
            mv = sb.http_post_json(sb.OT_GRAPHQL,
                                   {"query": "{ meta { apiVersion { x y z } } }"})
            v = mv["data"]["meta"]["apiVersion"]
            api_ver = f"{v['x']}.{v['y']}.{v['z']}"
        except Exception:
            pass

    AXES = {"overall": "overall", "clinical": "clinical",
            "genetic": "genetic_association"}
    rows = []
    pairs_AB = []   # overall vs clinical (headline)
    pairs_AC = []   # overall vs genetic
    pairs_BC = []   # clinical vs genetic
    n_generic = n_unscorable = 0

    for q in results:
        syms = q.get("_symbols", [])
        # union datatype rows across the query's target genes
        union = {ax: [] for ax in AXES}
        ok = bool(syms)
        for s in syms:
            dr = dt_cache.get(s)
            if dr is None:
                ok = False
                break
            for ax, axis_id in AXES.items():
                union[ax].extend(view(dr, axis_id))
        if not ok or not syms:
            n_unscorable += 1
            continue
        top_ents = [item["ent"] for item in q["top_k"][:TOP_K]]
        for ent in top_ents:
            tagA, otA, scA, gen = sb.classify_pred(ent, union["overall"], THRESHOLD)
            if gen:
                n_generic += 1
                continue
            tagB = sb.classify_pred(ent, union["clinical"], THRESHOLD)[0]
            tagC = sb.classify_pred(ent, union["genetic"], THRESHOLD)[0]
            rows.append({"query_idx": q["query_idx"], "genes": ";".join(syms),
                         "pred": ent,
                         "rater_A_overall": tagA, "rater_B_clinical": tagB,
                         "rater_C_genetic": tagC})
            pairs_AB.append((tagA, tagB))
            pairs_AC.append((tagA, tagC))
            pairs_BC.append((tagB, tagC))

    # per-prediction CSV
    out_csv = os.path.join(HERE, "second_rater_predictions.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_idx", "genes", "pred",
                                          "rater_A_overall", "rater_B_clinical",
                                          "rater_C_genetic"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    obs_AB, k_AB = cohen_kappa(pairs_AB)
    obs_AC, k_AC = cohen_kappa(pairs_AC)
    obs_BC, k_BC = cohen_kappa(pairs_BC)

    fracA = sum(1 for a, _ in pairs_AB if a == "PLAUSIBLE") / len(pairs_AB) if pairs_AB else None
    fracB = sum(1 for _, b in pairs_AB if b == "PLAUSIBLE") / len(pairs_AB) if pairs_AB else None
    fracC = sum(1 for _, c in pairs_AC if c == "PLAUSIBLE") / len(pairs_AC) if pairs_AC else None

    out = {
        "description": ("Independent AUTOMATED second rater (NOT a human). Two "
                        "Open Targets evidence axes score the SAME RotE top-20 "
                        "predictions at the SAME matched stringency (>=0.01) on "
                        "the SAME OT snapshot. Inter-rater agreement quantifies "
                        "how much the PLAUSIBLE/OFF-TARGET tag depends on which "
                        "evidence axis is privileged."),
        "is_human_rater": False,
        "ot_api_version_this_run": api_ver,
        "ot_api_version_primary_cache": "26.03 (gene_ot_assoc.json; headline enrichment)",
        "stringency_threshold": THRESHOLD,
        "top_k_scored": TOP_K,
        "model": "RotE/trial_101",
        "rater_definitions": {
            "rater_A_overall": "PRIMARY rater. OT OVERALL association score >= 0.01 (composite, harmonic-sum across all datatypes); identical rule to the headline analysis.",
            "rater_B_clinical": "HEADLINE second rater (independent automated, NOT human). OT 'clinical' DATATYPE score >= 0.01 -- clinical-precedence / known-drug evidence (ChEMBL drug-indication); a clinically meaningful axis distinct from the composite OVERALL score.",
            "rater_C_genetic": "Additional independent axis. OT 'genetic_association' DATATYPE score >= 0.01 -- human-genetics (GWAS/ClinVar) evidence; orthogonal but sparse for somatic cancers; stricter sensitivity comparator.",
        },
        "n_specific_predictions_tagged": len(rows),
        "n_generic_excluded": n_generic,
        "n_queries_unscorable": n_unscorable,
        "plausible_fraction": {"rater_A_overall": fracA,
                               "rater_B_clinical": fracB,
                               "rater_C_genetic": fracC},
        "agreement": {
            "primaryA_vs_clinicalB_HEADLINE": {
                "percent_agreement": obs_AB, "cohen_kappa": k_AB},
            "primaryA_vs_geneticC": {
                "percent_agreement": obs_AC, "cohen_kappa": k_AC},
            "clinicalB_vs_geneticC": {
                "percent_agreement": obs_BC, "cohen_kappa": k_BC},
        },
        "headline_second_rater": "rater_B_clinical",
        "headline_kappa": k_AB,
        "headline_percent_agreement": obs_AB,
    }
    out_json = os.path.join(HERE, "second_rater_agreement.json")
    sb.save_json(out_json, out)

    print(f"\ntagged {len(rows)} specific predictions; {n_generic} generic excluded; "
          f"{n_unscorable} queries unscorable")
    print(f"plausible-frac  A(overall)={fracA:.3f}  B(clinical)={fracB:.3f}  C(genetic)={fracC:.3f}")
    print(f"A vs B (overall vs clinical) HEADLINE: %agree={obs_AB:.3f}  kappa={k_AB:.3f}")
    print(f"A vs C (overall vs genetic):           %agree={obs_AC:.3f}  kappa={k_AC:.3f}")
    print(f"B vs C (clinical vs genetic):          %agree={obs_BC:.3f}  kappa={k_BC:.3f}")
    print(f"wrote {out_json}\nwrote {out_csv}")


if __name__ == "__main__":
    main()
