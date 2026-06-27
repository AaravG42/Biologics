#!/usr/bin/env python3
"""
external_validation.py
======================

EXTERNAL, evidence-held-out validation of the RotE KG-embedding model for
monoclonal-antibody (mAb) repurposing.

WHY THIS IS NOT A TEMPORAL SPLIT  (read first)
----------------------------------------------
A reviewer (in the spirit of Briere et al.) asked for a genuinely *external*
inference set rather than a same-KG random test split.  The strongest such set
is a *temporally* held-out split (train on what was known before year Y, test on
what became known after).  THAT IS IMPOSSIBLE WITH THE PROVIDED DATA: the
IMGT_no_leakage_directional_clinical knowledge graph contains NO timestamps.  We
verified this directly --- the only date-bearing content is (a)
`imgt:hasStatut imgt:approval` (a status flag carrying no date) and (b) DOIs
inside `imgt:hasBibliographicReference` whose strings incidentally contain a
publication year; there is NO date / year / approval-year relation or entity
tying an *indication* to a point in time.  A strict temporally held-out
evaluation therefore cannot be constructed from these data.

WHAT WE DO INSTEAD: EXTERNAL-EVIDENCE HELD-OUT SET
--------------------------------------------------
  1. For every forward query (StudyProduct, imgt:hasClinicalIndication, ?) the
     RotE model was evaluated on, resolve the mAb's protein target:
         StudyProduct --isStudyProductOf--> Product --isProductOf--> mAb
                      --(sio:SIO_000291 | isTargetOf)--> HGNC gene
     HGNC -> symbol (MyGene.info, cached) -> Ensembl id (Open Targets, cached).

  2. Pull EXTERNAL clinical-trial evidence for that target from Open Targets
     `target.drugAndClinicalCandidates` (successor of `knownDrugs`).  Open
     Targets aggregates ChEMBL / ClinicalTrials.gov (AACT) / FDA / EMA.  For
     each ANTIBODY drug against the target we collect the indications studied
     plus the ClinicalTrials.gov NCT ids and per-trial phase.  All API calls are
     cached to disk; access date recorded.

     ClinicalTrials.gov-by-name is INFEASIBLE here: the KG ANONYMISES antibody
     identity (entities are `imgt:mAb_<id>`; 0 INN tokens exist in the graph).
     The molecular TARGET is the only stable bridge between the anonymised KG
     mAbs and external clinical evidence, so the evidence is TARGET-LEVEL: it
     shows an antibody against this mAb's target is in clinical trials for D (it
     may be a different antibody against the same target).  Stated in METHODS.md.

  3. Map each external Open Targets EFO disease to an IMGT KG indication entity
     with a transparent normalisation map (reused from
     revision/blinded_plausibility/score_blinded.py).  Report coverage.

  4. Keep only HELD-OUT pairs: (mAb, indication D) is held-out iff D is NOT a
     TRAINING `imgt:hasClinicalIndication` tail for ANY study product of that
     mAb.  Generic umbrella indications (Solid_tumors, Cancers, ...) are EXCLUDED
     so the test is not trivially won by always naming the most popular bucket.

  5. Evaluate with the model's AUTHORITATIVE per-query predictions in
     interpret_topk_50.json.  (NB: the model.pt currently in the trial dir is a
     *later* hard-negative finetune (mean_rank 24.56); interpret_topk_50.json is
     the pre-finetune checkpoint (mean_rank 24.96) and is the published ranking
     this analysis is asked to use -- so we read ranks from the JSON rather than
     re-scoring with the mismatched checkpoint.)  For the head-to-head against
     the popularity prior, both methods rank D within the SAME 247-indication
     pool (apples-to-apples).  enrichment = model_recall@k / prior_recall@k.

OUTPUTS (next to this script):  external_pairs.csv, summary.json, cache/.
CPU ONLY.  Run:  CUDA_VISIBLE_DEVICES="" python3 external_validation.py
"""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU only (no torch is used anyway)

import sys
import csv
import json
import re
import time
import datetime
import argparse
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BLINDED = os.path.join(REPO, "revision", "blinded_plausibility")

sys.path.insert(0, REPO)
sys.path.insert(0, BLINDED)
os.environ.setdefault("DATA_PATH", os.path.join(REPO, "data"))

import score_blinded as sb  # noqa: E402  (transparent reuse of normalisation/KG logic)

# --------------------------------------------------------------------------- #
#  PATHS                                                                       #
# --------------------------------------------------------------------------- #

TRIAL_DIR = os.path.join(
    REPO, "analysis_outputs", "IMGT_no_leakage_directional_clinical",
    "clinical_hpo", "RotE", "trial_101")
INTERPRET_JSON = os.path.join(TRIAL_DIR, "interpret_topk_50.json")
PRIOR_JSON = os.path.join(TRIAL_DIR, "indication_prior.json")
KG_DIR = os.path.join(REPO, "data", "IMGT_no_leakage_directional_clinical")

CACHE_DIR = os.path.join(HERE, "cache")
HGNC_CACHE = os.path.join(CACHE_DIR, "hgnc2symbol.json")
ENSEMBL_CACHE = os.path.join(CACHE_DIR, "symbol2ensembl.json")
OT_DAC_CACHE = os.path.join(CACHE_DIR, "ot_drug_candidates.json")
META_CACHE = os.path.join(CACHE_DIR, "fetch_meta.json")

OUT_CSV = os.path.join(HERE, "external_pairs.csv")
OUT_SUMMARY = os.path.join(HERE, "summary.json")

OT_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"
CLINICAL_REL = "imgt:hasClinicalIndication"
RECALL_KS = [5, 10, 20]
TOPK_DEPTH = 50   # interpret_topk_50.json stores 50 predictions per query


# --------------------------------------------------------------------------- #
#  Open Targets external clinical-trial evidence                               #
# --------------------------------------------------------------------------- #

DAC_QUERY = """
query DAC($ens: String!) {
  target(ensemblId: $ens) {
    approvedSymbol
    drugAndClinicalCandidates {
      count
      rows {
        maxClinicalStage
        drug { id name drugType }
        diseases { disease { id name } }
        clinicalReports { id source url trialPhase diseases { disease { id } } }
      }
    }
  }
}"""


def parse_phase(s):
    """Map an Open Targets phase/stage string to a numeric phase (0-4)."""
    if not s:
        return 0
    u = s.upper()
    if "APPROV" in u or "PHASE_4" in u or "PHASE4" in u:
        return 4
    nums = [int(x) for x in re.findall(r"[1-4]", u)]
    return max(nums) if nums else 0


def is_nct(report_id, url):
    rid = (report_id or "").strip().lower()
    return rid.startswith("nct") or "clinicaltrials.gov" in (url or "").lower()


def is_antibody(drug_type, drug_name):
    dt = (drug_type or "").lower()
    return ("antibody" in dt) or (drug_name or "").lower().endswith("mab")


def fetch_ot_drug_candidates(ensembl_id, symbol, cache, meta, live):
    """Fetch + cache external clinical-trial evidence for one target gene."""
    if ensembl_id in cache:
        return cache[ensembl_id]
    if not live:
        return None
    try:
        res = sb.http_post_json(OT_GRAPHQL,
                                {"query": DAC_QUERY, "variables": {"ens": ensembl_id}})
    except Exception as e:  # noqa: BLE001
        print(f"  [OT] {symbol}/{ensembl_id} FAILED: {e}")
        return None
    tgt = ((res.get("data") or {}).get("target")) or {}
    dac = tgt.get("drugAndClinicalCandidates") or {}
    rows_out = []
    for row in dac.get("rows", []) or []:
        drug = row.get("drug") or {}
        diseases = [[(d.get("disease") or {}).get("id"),
                     ((d.get("disease") or {}).get("name") or "").strip().lower()]
                    for d in (row.get("diseases") or [])
                    if (d.get("disease") or {}).get("id")]
        ncts = []
        for rep in row.get("clinicalReports", []) or []:
            rid = rep.get("id")
            if not is_nct(rid, rep.get("url")):
                continue
            ph = parse_phase(rep.get("trialPhase"))
            for d in rep.get("diseases", []) or []:
                eid = (d.get("disease") or {}).get("id")
                if eid:
                    ncts.append([eid, rid.upper(), rep.get("source") or "", ph])
        rows_out.append({"drug": (drug.get("name") or "").strip(),
                         "drugType": (drug.get("drugType") or "").strip(),
                         "maxStage": row.get("maxClinicalStage"),
                         "diseases": diseases, "ncts": ncts})
    cache[ensembl_id] = {"symbol": symbol, "count": dac.get("count", 0),
                         "rows": rows_out}
    meta[ensembl_id] = {"symbol": symbol,
                        "accessed_utc": datetime.datetime.now(datetime.timezone.utc)
                        .isoformat(), "n_rows": len(rows_out)}
    sb.save_json(OT_DAC_CACHE, cache)
    sb.save_json(META_CACHE, meta)
    time.sleep(sb.API_SLEEP)
    return cache[ensembl_id]


# --------------------------------------------------------------------------- #
#  OT EFO disease name -> IMGT KG indication entity                            #
# --------------------------------------------------------------------------- #

def build_kg_indication_index(indication_entities):
    index = []
    for ent in indication_entities:
        core, keywords, is_generic, _ = sb.normalise_disease(ent)
        toks = set()
        for kw in keywords:
            toks |= sb._tokens(kw)
        index.append({"ent": ent, "core": core, "keywords": set(keywords),
                      "tokens": toks, "is_generic": is_generic})
    return index


def normalise_ot_to_kg(ot_name, kg_index, prior_counts):
    """Map an OT EFO disease name to the best KG indication entity (or None).
    1) exact normalised-phrase match; 2) discriminative-token overlap (generic
    tokens excluded, see score_blinded.STOP_TOKENS), best overlap tie-broken by
    KG training prior then shorter entity.  Generic KG nodes are not eligible.
    """
    clean = sb._clean(ot_name)
    ot_tokens = sb._tokens(clean)
    for c in kg_index:
        if c["is_generic"]:
            continue
        if clean == c["core"] or clean in c["keywords"]:
            return c["ent"], "exact"
    if not ot_tokens:
        return None, "no_discriminative_token"
    best = None
    for c in kg_index:
        if c["is_generic"]:
            continue
        shared = ot_tokens & c["tokens"]
        if not shared:
            continue
        key = (len(shared), prior_counts.get(c["ent"], 0), -len(c["ent"]))
        if best is None or key > best[0]:
            best = (key, c["ent"])
    return (best[1], "token_overlap") if best else (None, "unmatched")


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #

def metrics_over(rows, rank_key, pool_size):
    """rank_key None/0 => censored (beyond stored depth) => counts as a miss for
    recall and is excluded from median (with censoring reported)."""
    ranks = [r[rank_key] for r in rows]
    present = [x for x in ranks if x is not None]
    out = {"n_pairs": len(rows), "n_ranked_within_depth": len(present),
           "frac_censored_beyond_depth": (1 - len(present) / len(rows)) if rows else None}
    for k in RECALL_KS:
        out[f"recall@{k}"] = (sum(1 for x in present if x <= k) / len(rows)
                              if rows else None)
    out["median_rank_within_depth"] = float(np.median(present)) if present else None
    out["mrr"] = (sum((1.0 / x) for x in present) / len(rows)) if rows else None
    return out


def _hit(rank, k):
    return 1 if (rank is not None and rank <= k) else 0


def bootstrap_recall_ci(rows, k, n_boot=2000, seed=20240617):
    """Paired bootstrap over pairs: returns CI for model recall@k, prior
    recall@k and the enrichment ratio (model/prior)."""
    if not rows:
        return None
    rng = np.random.default_rng(seed)
    m = np.array([_hit(r["model_rank_indication_pool"], k) for r in rows])
    p = np.array([_hit(r["prior_rank_indication_pool"], k) for r in rows])
    n = len(rows)
    mr, pr, ra = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        mm, pp = m[idx].mean(), p[idx].mean()
        mr.append(mm); pr.append(pp)
        ra.append(mm / pp if pp > 0 else np.nan)
    def ci(a):
        a = np.array(a, dtype=float)
        a = a[~np.isnan(a)]
        return [round(float(np.percentile(a, 2.5)), 3),
                round(float(np.percentile(a, 97.5)), 3)] if len(a) else None
    return {"model_recall_ci95": ci(mr), "prior_recall_ci95": ci(pr),
            "enrichment_ci95": ci(ra)}


def enrichment_block(rows, pool_size):
    model = metrics_over(rows, "model_rank_indication_pool", pool_size)
    baseline = metrics_over(rows, "prior_rank_indication_pool", pool_size)
    enr, boot = {}, {}
    for k in RECALL_KS:
        mk, bk = model[f"recall@{k}"], baseline[f"recall@{k}"]
        enr[f"recall@{k}"] = round(mk / bk, 3) if (bk) else None
        boot[f"recall@{k}"] = bootstrap_recall_ci(rows, k)
    return {"n_pairs": len(rows), "model": model,
            "popularity_prior_baseline": baseline,
            "enrichment_model_over_prior": enr,
            "bootstrap_ci95": boot}


def dedup_by_mab(rows):
    """Unique (mAb, indication); keep the best (smallest) model rank.
    Censored (None) sorts after any finite rank."""
    best = {}
    for r in rows:
        key = (r["mab"], r["external_indication_kg"])
        cur = r["model_rank_indication_pool"]
        cur = cur if cur is not None else 10 ** 9
        if key not in best:
            best[key] = r
        else:
            prev = best[key]["model_rank_indication_pool"]
            prev = prev if prev is not None else 10 ** 9
            if cur < prev:
                best[key] = r
    return list(best.values())


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true",
                    help="Use only on-disk caches; do not hit any API.")
    args = ap.parse_args()
    live = not args.offline
    os.makedirs(CACHE_DIR, exist_ok=True)
    run_date = datetime.date.today().isoformat()

    interp = json.load(open(INTERPRET_JSON))
    results = interp["results"]
    prior_counts = json.load(open(PRIOR_JSON))["counts"]

    head_index, tail_index = sb.load_kg()
    indication_entities = sorted({
        t for _h, rels in head_index.items() for (r, t) in rels if r == CLINICAL_REL})
    pool_set = set(indication_entities)
    pool_size = len(indication_entities)

    # per-StudyProduct training (and valid/test) indications
    train_ind_by_sp, vt_ind_by_sp = defaultdict(set), defaultdict(set)
    for split in ("train", "valid", "test"):
        path = os.path.join(KG_DIR, split)
        if not os.path.exists(path):
            continue
        for line in open(path):
            p = line.rstrip("\n").split("\t")
            if len(p) == 3 and p[1] == CLINICAL_REL:
                (train_ind_by_sp if split == "train" else vt_ind_by_sp)[p[0]].add(p[2])

    sp_to_mab, mab_to_sps = {}, defaultdict(set)
    for h in list(head_index.keys()):
        if h.startswith("imgt:StudyProduct"):
            _p, mab = sb.resolve_mab(h, head_index)
            if mab:
                sp_to_mab[h] = mab
                mab_to_sps[mab].add(h)

    def mab_train_inds(mab):
        out = set()
        for sp in mab_to_sps.get(mab, ()):
            out |= train_ind_by_sp.get(sp, set())
        return out

    # resolve target genes per query
    hgnc_cache = sb.load_json(HGNC_CACHE, {})
    ensembl_cache = sb.load_json(ENSEMBL_CACHE, {})
    query_info, needed_ens = [], {}
    for row in results:
        sp = row["head"]
        _p, mab = sb.resolve_mab(sp, head_index)
        symbols, ens_ids = [], []
        for hg in sb.resolve_targets(mab, head_index, tail_index):
            sym = sb.hgnc_to_symbol(hg, hgnc_cache, live)
            if not sym:
                continue
            symbols.append(sym)
            ens = sb.symbol_to_ensembl(sym, ensembl_cache, live)
            if ens:
                ens_ids.append(ens)
                needed_ens[ens] = sym
        # ordered indication-only ranking from the authoritative top-k
        ind_rank = {}
        pos = 0
        for item in row["top_k"]:
            if item["ent"] in pool_set:
                pos += 1
                ind_rank.setdefault(item["ent"], pos)
        query_info.append({
            "query_idx": row["query_idx"], "study_product": sp, "mab": mab,
            "symbols": symbols, "ensembl": ens_ids, "tail_true": row["tail_true"],
            "topk_pos": {it["ent"]: it["pos"] for it in row["top_k"]},
            "ind_rank": ind_rank})
    sb.save_json(HGNC_CACHE, hgnc_cache)
    sb.save_json(ENSEMBL_CACHE, ensembl_cache)

    # fetch external evidence per gene
    ot_cache = sb.load_json(OT_DAC_CACHE, {})
    meta = sb.load_json(META_CACHE, {})
    print(f"Open Targets clinical candidates for {len(needed_ens)} genes (live={live}) ...")
    for i, (ens, sym) in enumerate(sorted(needed_ens.items()), 1):
        if ens not in ot_cache:
            fetch_ot_drug_candidates(ens, sym, ot_cache, meta, live)
            if i % 20 == 0:
                print(f"  {i}/{len(needed_ens)}")

    # gene -> {efo: {name, drugs, ncts, max_phase}}  (ANTIBODY rows only)
    gene_ext, all_ot_names = {}, set()
    for ens, payload in ot_cache.items():
        if not payload:
            continue
        per = {}
        for r in payload.get("rows", []):
            if not is_antibody(r.get("drugType"), r.get("drug")):
                continue
            phase_by_efo = defaultdict(int)
            nct_by_efo = defaultdict(set)
            for efo, nct, _src, ph in r.get("ncts", []):
                nct_by_efo[efo].add(nct)
                phase_by_efo[efo] = max(phase_by_efo[efo], ph)
            for efo, name in r.get("diseases", []):
                all_ot_names.add(name)
                d = per.setdefault(efo, {"name": name, "drugs": set(),
                                         "ncts": set(), "max_phase": 0})
                d["drugs"].add(r.get("drug"))
                d["ncts"] |= nct_by_efo.get(efo, set())
                d["max_phase"] = max(d["max_phase"], phase_by_efo.get(efo, 0))
        gene_ext[ens] = per

    # normalise distinct OT disease names once
    kg_index = build_kg_indication_index(indication_entities)
    name_to_kg = {n: normalise_ot_to_kg(n, kg_index, prior_counts)
                  for n in sorted(all_ot_names)}
    n_names_matched = sum(1 for v in name_to_kg.values() if v[0])

    # popularity-prior ranking within the indication pool (competition rank)
    pool_counts = {e: prior_counts.get(e, 0) for e in indication_entities}
    def prior_rank(ent):
        c = pool_counts.get(ent, 0)
        return 1 + sum(1 for x in pool_counts.values() if x > c)
    prior_rank_cache = {e: prior_rank(e) for e in indication_entities}

    # build external pairs
    rows_csv = []
    for qi in query_info:
        if not qi["ensembl"]:
            continue
        mab = qi["mab"]
        train_inds = mab_train_inds(mab)
        vt_inds = set()
        for sp in mab_to_sps.get(mab, ()):
            vt_inds |= vt_ind_by_sp.get(sp, set())
        ext = {}
        for ens in qi["ensembl"]:
            for efo, info in gene_ext.get(ens, {}).items():
                kg_ent, _kind = name_to_kg.get(info["name"], (None, None))
                if kg_ent is None:
                    continue
                e = ext.setdefault(kg_ent, {"drugs": set(), "ncts": set(),
                                            "efo": set(), "names": set(),
                                            "max_phase": 0})
                e["drugs"] |= info["drugs"]
                e["ncts"] |= info["ncts"]
                e["efo"].add(efo)
                e["names"].add(info["name"])
                e["max_phase"] = max(e["max_phase"], info["max_phase"])
        for kg_ent, e in ext.items():
            rows_csv.append({
                "query_idx": qi["query_idx"], "study_product": qi["study_product"],
                "mab": mab, "target_genes": "|".join(qi["symbols"]),
                "external_indication_kg": kg_ent,
                "external_indication_efo": ";".join(sorted(e["names"]))[:300],
                "efo_ids": "|".join(sorted(e["efo"])),
                "source": "OpenTargets:drugAndClinicalCandidates(ChEMBL/ClinicalTrials.gov)",
                "evidence_drugs": "|".join(sorted(x for x in e["drugs"] if x))[:300],
                "nct_ids": "|".join(sorted(e["ncts"])[:25]), "n_ncts": len(e["ncts"]),
                "max_trial_phase": e["max_phase"],
                "in_training": kg_ent in train_inds,
                "in_valid_test": kg_ent in vt_inds,
                "model_rank_indication_pool": qi["ind_rank"].get(kg_ent),
                "model_rank_all_entities": qi["topk_pos"].get(kg_ent),
                "prior_rank_indication_pool": prior_rank_cache.get(kg_ent),
            })

    # ----- subsets ------------------------------------------------------
    heldout = [r for r in rows_csv if not r["in_training"]]
    heldout_nct = [r for r in heldout if r["n_ncts"] > 0]
    heldout_nct_p2 = [r for r in heldout_nct if r["max_trial_phase"] >= 2]
    corroborated = [r for r in heldout_nct if r["in_valid_test"]]  # KG+external agree

    summary = {
        "title": "External-evidence held-out validation (RotE, mAb repurposing)",
        "run_date": run_date, "cpu_only": True,
        "no_timestamp_limitation": (
            "The KG has NO timestamps; a strict temporally-held-out split is "
            "INFEASIBLE from the provided data. Only date-bearing content is "
            "imgt:hasStatut=approval (no date) and DOIs in bibliographic refs "
            "(incidental years). We therefore use an EXTERNAL-EVIDENCE held-out "
            "set, not a temporal one."),
        "model_predictions_used": INTERPRET_JSON,
        "model_checkpoint_note": (
            "Ranks read from interpret_topk_50.json (pre-finetune checkpoint, "
            "test mean_rank=24.96 = the published ranking). model.pt in the trial "
            "dir is a later hard-negative finetune (mean_rank=24.56) and does NOT "
            "reproduce the published top-k, so it is intentionally not used."),
        "external_source": {
            "name": "Open Targets Platform GraphQL target.drugAndClinicalCandidates",
            "endpoint": OT_GRAPHQL,
            "aggregates": "ChEMBL / ClinicalTrials.gov (AACT) / FDA / EMA",
            "bridge": "molecular target gene (KG anonymises antibody identity; no "
                      "INN entities -> ClinicalTrials.gov-by-name infeasible). "
                      "Evidence is TARGET-LEVEL.",
            "access_dates": sorted({(m.get("accessed_utc") or "")[:10]
                                    for m in meta.values() if m}),
        },
        "candidate_pool_size_indications": pool_size,
        "resolution": {
            "n_queries": len(query_info),
            "n_queries_with_target": sum(1 for q in query_info if q["ensembl"]),
            "n_unique_genes": len(needed_ens)},
        "normalisation_coverage": {
            "n_distinct_ot_disease_names": len(all_ot_names),
            "n_ot_names_mapped_to_kg_entity": n_names_matched,
            "coverage_fraction": round(n_names_matched / len(all_ot_names), 4)
            if all_ot_names else None,
            "note": "Most OT EFO diseases have no counterpart among the "
                    f"{pool_size} KG indications; coverage = fraction of distinct "
                    "OT disease strings mapping to a (specific, non-generic) KG node."},
        "counts": {
            "external_pairs_total": len(rows_csv),
            "heldout_not_in_training": len(heldout),
            "heldout_with_ctgov_nct": len(heldout_nct),
            "heldout_with_nct_phase2plus": len(heldout_nct_p2),
            "heldout_externally_AND_kg_corroborated(in_valid_test)": len(corroborated),
            "unique_mab_indication_heldout_nct": len(dedup_by_mab(heldout_nct)),
        },
        "results": {
            "A_heldout_nct_unique_mab_indication":
                enrichment_block(dedup_by_mab(heldout_nct), pool_size),
            "B_heldout_nct_phase2plus_unique_mab_indication":
                enrichment_block(dedup_by_mab(heldout_nct_p2), pool_size),
            "C_externally_corroborated_test_indications_unique_mab_indication":
                enrichment_block(dedup_by_mab(corroborated), pool_size),
            "D_heldout_nct_per_query":
                enrichment_block(heldout_nct, pool_size),
        },
        "random_uniform_recall_reference": {
            f"recall@{k}": round(k / pool_size, 4) for k in RECALL_KS},
        "interpretation": (
            "recall@k = fraction of held-out externally-evidenced (mAb,indication) "
            "pairs for which the model ranks the indication in its top-k (within "
            "the 247-indication pool). enrichment = model recall / popularity-prior "
            "recall. Set C (model's own held-out test indications independently "
            "corroborated by external ClinicalTrials.gov evidence) is the strongest "
            "positive set; Sets A/B test the broader target-level external space."),
    }

    fieldnames = ["query_idx", "study_product", "mab", "target_genes",
                  "external_indication_kg", "external_indication_efo", "efo_ids",
                  "source", "evidence_drugs", "nct_ids", "n_ncts",
                  "max_trial_phase", "in_training", "in_valid_test",
                  "model_rank_indication_pool", "model_rank_all_entities",
                  "prior_rank_indication_pool"]
    rows_csv.sort(key=lambda r: (r["in_training"],
                                 r["model_rank_indication_pool"] is None,
                                 r["model_rank_indication_pool"] or 10 ** 9))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_csv:
            w.writerow({k: r.get(k) for k in fieldnames})
    sb.save_json(OUT_SUMMARY, summary)

    # console report
    print("\n=== EXTERNAL-EVIDENCE HELD-OUT VALIDATION ===")
    nc = summary["normalisation_coverage"]
    print(f"distinct OT disease names: {nc['n_distinct_ot_disease_names']}  "
          f"mapped to KG: {nc['n_ot_names_mapped_to_kg_entity']} "
          f"({100*nc['coverage_fraction']:.1f}%)")
    c = summary["counts"]
    print(f"external pairs total={c['external_pairs_total']}  "
          f"held-out(not train)={c['heldout_not_in_training']}  "
          f"+CT.gov NCT={c['heldout_with_ctgov_nct']}  "
          f"+phase>=2={c['heldout_with_nct_phase2plus']}  "
          f"externally+KG corroborated={c['heldout_externally_AND_kg_corroborated(in_valid_test)']}")
    for tag, key in [("A heldout+NCT (uniq mAb x ind)", "A_heldout_nct_unique_mab_indication"),
                     ("B heldout+NCT phase>=2", "B_heldout_nct_phase2plus_unique_mab_indication"),
                     ("C ext+KG corroborated", "C_externally_corroborated_test_indications_unique_mab_indication")]:
        blk = summary["results"][key]
        m, b, e = blk["model"], blk["popularity_prior_baseline"], blk["enrichment_model_over_prior"]
        print(f"\n[{tag}] n={blk['n_pairs']}")
        for k in RECALL_KS:
            print(f"  recall@{k}: model={m[f'recall@{k}']:.3f}  prior={b[f'recall@{k}']:.3f}  enr={e[f'recall@{k}']}x")
        print(f"  median rank within top-{TOPK_DEPTH}: model={m['median_rank_within_depth']}  "
              f"prior={b['median_rank_within_depth']}  "
              f"(model censored beyond depth: {100*m['frac_censored_beyond_depth']:.0f}%)")
    print(f"\nWrote: {OUT_CSV}\n       {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
