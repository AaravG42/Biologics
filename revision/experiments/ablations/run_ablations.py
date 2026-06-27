"""Rigorous ablation analyses on already-trained IMGT clinical KG-embedding models.

Runs on CPU only. Reuses the repo's model/dataset machinery.

Three analyses (see ABLATIONS.md):
  (1) Distractor-removal ablation (RotE, TransE; esm2; 5 seeds).
  (2) Per-query excluded-node-as-ground-truth accounting.
  (3) 483/201 stratified ESM vs random delta-MRR.

Outputs (csv) written next to this script.
"""
import os, sys, json, csv, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("DATA_PATH", "/home/aarav/KGEmb/data")
REPO = "/home/aarav/KGEmb"
sys.path.insert(0, REPO)

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

import models
from datasets.kg_dataset import KGDataset
from datasets.process import get_idx
from utils.imgt_clinical import CLINICAL_INDICATION_RELATION, MAB_PATTERN

# ---------------------------------------------------------------------------
# CPU fallback (mirror clinical_eval.enable_cpu_fallback)
# ---------------------------------------------------------------------------
def enable_cpu_fallback():
    if torch.cuda.is_available():
        return "cuda"
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self
    return "cpu"

DEVICE = enable_cpu_fallback()

DATA_NAME = "IMGT_no_leakage_directional_clinical"
DATA_DIR = Path(REPO) / "data" / DATA_NAME
OUT_DIR = Path(REPO) / "revision" / "experiments" / "ablations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXP = Path(REPO) / "revision" / "experiments"
MODEL_GROUP = {
    "RotE": "runs_g0", "TransE": "runs_g0", "RefE": "runs_g0",
    "MurE": "runs_g1", "AttE": "runs_g1", "CP": "runs_g1",
    "AttH": "runs_g3", "RefH": "runs_g3", "RotH": "runs_g3",
}
SEEDS = [0, 1, 2, 3, 4]

TOP5 = [
    "imgt:Solid_tumors",
    "imgt:Cancers_non-small_cell_lung_(NSCLC)",
    "imgt:Cancers",
    "imgt:Multiple_myeloma_(MM)",
    "imgt:Non-Hodgkins_lymphoma_(NHL)",
]

# ---------------------------------------------------------------------------
# Dataset-level structures (shared)
# ---------------------------------------------------------------------------
ent2idx, rel2idx = get_idx(str(DATA_DIR))
idx2ent = {i: e for e, i in ent2idx.items()}
FWD_REL = rel2idx[CLINICAL_INDICATION_RELATION]
assert FWD_REL == 6, FWD_REL


def load_raw_split(split):
    out = []
    with open(DATA_DIR / split) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            h, r, t = line.split("\t")
            out.append((h, r, t))
    return out

TRAIN = load_raw_split("train")
VALID = load_raw_split("valid")
TEST = load_raw_split("test")
ALL = TRAIN + VALID + TEST

FWD_TEST = [(h, r, t) for (h, r, t) in TEST if r == CLINICAL_INDICATION_RELATION]
assert len(FWD_TEST) == 297, len(FWD_TEST)

# train forward tail frequency (for top-5 verification + random-pool freqs)
TRAIN_TAIL_FREQ = Counter(t for (h, r, t) in TRAIN if r == CLINICAL_INDICATION_RELATION)
ALL_TAIL_FREQ = Counter(t for (h, r, t) in ALL if r == CLINICAL_INDICATION_RELATION)

# disease tail pool (all entities appearing as forward tails, any split)
DISEASE_TAILS = sorted(set(t for (h, r, t) in ALL if r == CLINICAL_INDICATION_RELATION))

# studyproduct -> mab traversal
_sp2prod, _prod2mab = {}, {}
for h, r, t in ALL:
    if r == "imgt:isStudyProductOf":
        _sp2prod[h] = t
    elif r == "imgt:isProductOf":
        _prod2mab[h] = t
SP2MAB = {sp: _prod2mab[p] for sp, p in _sp2prod.items() if p in _prod2mab}

# ESM init status
ESM_INIT = np.load("/home/aarav/Biologics/esm_init/entity_init.npy")
ESM_NORM = np.linalg.norm(ESM_INIT, axis=1)
ESM_INIT_IDS = set(int(i) for i in np.where(ESM_NORM > 0.3)[0])  # 483 mAbs

# per-query mab + esm status for the 297 forward test triples
QUERY_HEAD = [ent2idx[h] for (h, r, t) in FWD_TEST]
QUERY_TAIL = [ent2idx[t] for (h, r, t) in FWD_TEST]
QUERY_TAIL_NAME = [t for (h, r, t) in FWD_TEST]
QUERY_MAB = []
QUERY_STRATUM = []  # "esm" / "random" / "unresolved"
for (h, r, t) in FWD_TEST:
    mab = SP2MAB.get(h)
    QUERY_MAB.append(mab)
    if mab is None:
        QUERY_STRATUM.append("unresolved")
    else:
        mid = ent2idx.get(mab)
        QUERY_STRATUM.append("esm" if (mid is not None and mid in ESM_INIT_IDS) else "random")

# shared dataset object (filters) - build once
DATASET = KGDataset(str(DATA_DIR), False)
RHS_FILTERS = DATASET.get_filters()["rhs"]
N_ENT = DATASET.get_shape()[0]


# ---------------------------------------------------------------------------
# Model loading (config embedded in metrics.json)
# ---------------------------------------------------------------------------
def load_model_dir(model_dir):
    md = json.loads((Path(model_dir) / "metrics.json").read_text())
    config = dict(md["config"])
    config.setdefault("sizes", DATASET.get_shape())
    config.setdefault("dropout", 0.0)
    config.setdefault("gamma", 12.0)
    config.setdefault("dtype", "single")
    config.setdefault("bias", "constant")
    config.setdefault("init_size", 1e-3)
    config.setdefault("rank", 640)
    config.setdefault("multi_c", False)
    args = argparse.Namespace(**config)
    model = getattr(models, args.model)(args)
    model.to(DEVICE)
    state = torch.load(Path(model_dir) / "model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def base_filtered_scores(model):
    """Return (scores[297,N_ENT], targets[297]) with standard rhs-filters applied
    (matching KGModel.get_ranking semantics: filtered tails incl. gold set to -1e6).
    """
    queries = torch.tensor(
        [[QUERY_HEAD[i], FWD_REL, QUERY_TAIL[i]] for i in range(len(FWD_TEST))],
        dtype=torch.long, device=DEVICE,
    )
    with torch.no_grad():
        q = model.get_queries(queries)
        cands = model.get_rhs(queries, eval_mode=True)
        rhs = model.get_rhs(queries, eval_mode=False)
        scores = model.score(q, cands, eval_mode=True).float().cpu().numpy()  # [297, N_ENT]
        targets = model.score(q, rhs, eval_mode=False).float().cpu().numpy().reshape(-1)  # [297]
    # apply standard filters
    for i in range(len(FWD_TEST)):
        key = (QUERY_HEAD[i], FWD_REL)
        filt = list(RHS_FILTERS.get(key, ()))
        tgt = QUERY_TAIL[i]
        if tgt not in filt:
            filt.append(tgt)
        scores[i, filt] = -1e6
    return scores, targets


def ranks_from_scores(scores, targets, exclude_ids=None, gold_excluded_policy="inf"):
    """Compute per-query filtered ranks. exclude_ids: set of entity ids to drop
    from candidate pool (their score -> -1e6). For queries whose gold tail is in
    exclude_ids, set rank=inf (policy='inf')."""
    n = scores.shape[0]
    s = scores
    if exclude_ids:
        ex = np.array(sorted(exclude_ids), dtype=np.int64)
        s = scores.copy()
        s[:, ex] = -1e6
    ranks = np.empty(n, dtype=np.float64)
    ex_set = set(exclude_ids) if exclude_ids else set()
    for i in range(n):
        if QUERY_TAIL[i] in ex_set:
            ranks[i] = math.inf
            continue
        ranks[i] = 1.0 + float(np.sum(s[i] >= targets[i]))
    return ranks


def metrics_from_ranks(ranks, subset=None):
    """MRR/H@1/H@10 over ranks (inf -> 0 contribution). subset: index array."""
    r = ranks if subset is None else ranks[subset]
    if len(r) == 0:
        return {"n": 0, "MRR": float("nan"), "H1": float("nan"), "H10": float("nan")}
    inv = np.where(np.isinf(r), 0.0, 1.0 / r)
    return {
        "n": int(len(r)),
        "MRR": float(np.mean(inv)),
        "H1": float(np.mean((r <= 1).astype(float))),
        "H10": float(np.mean((r <= 10).astype(float))),
    }


# ===========================================================================
# ANALYSIS 2 (pure data accounting)
# ===========================================================================
def analysis2():
    rows = []
    top5set = set(TOP5)
    cnt = Counter(QUERY_TAIL_NAME[i] for i in range(len(FWD_TEST)) if QUERY_TAIL_NAME[i] in top5set)
    total = sum(cnt.values())
    for name in TOP5:
        rows.append({
            "excluded_tail": name,
            "entity_id": ent2idx[name],
            "train_fwd_freq": TRAIN_TAIL_FREQ.get(name, 0),
            "all_fwd_freq": ALL_TAIL_FREQ.get(name, 0),
            "num_test_queries_with_this_gold": cnt.get(name, 0),
        })
    rows.append({
        "excluded_tail": "TOTAL",
        "entity_id": "",
        "train_fwd_freq": sum(TRAIN_TAIL_FREQ.get(n, 0) for n in TOP5),
        "all_fwd_freq": sum(ALL_TAIL_FREQ.get(n, 0) for n in TOP5),
        "num_test_queries_with_this_gold": total,
    })
    with open(OUT_DIR / "excluded_as_gold.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "excluded_tail", "entity_id", "train_fwd_freq", "all_fwd_freq",
            "num_test_queries_with_this_gold"])
        w.writeheader()
        w.writerows(rows)
    print(f"[A2] {total}/297 test queries have a TOP5 gold tail. Wrote excluded_as_gold.csv")
    return total


# ===========================================================================
# ANALYSIS 1 (distractor-removal ablation)  RotE, TransE; esm2; 5 seeds
# ===========================================================================
def analysis1(n_draws=50, rng_seed=12345):
    rng = np.random.default_rng(rng_seed)
    top5_ids = set(ent2idx[n] for n in TOP5)
    # random pool: disease tail entity ids NOT in top5
    pool = [ent2idx[d] for d in DISEASE_TAILS if d not in set(TOP5)]
    pool = np.array(sorted(pool), dtype=np.int64)
    # pre-generate the same random draws for all models/seeds for comparability
    draws = [set(int(x) for x in rng.choice(pool, size=5, replace=False)) for _ in range(n_draws)]

    rows = []
    for model_name in ["RotE", "TransE"]:
        group = MODEL_GROUP[model_name]
        per_seed = {"std": [], "freq": [], "freq_unaff": [], "std_on_freq_unaff": [],
                    "rand": defaultdict(list), "rand_unaff": defaultdict(list),
                    "std_on_rand_unaff": defaultdict(list), "rand_n_goldex": []}
        for seed in SEEDS:
            md = EXP / group / model_name / "esm2" / f"seed_{seed}"
            if not (md / "model.pt").exists():
                print(f"  [A1] MISSING {md}")
                continue
            model = load_model_dir(md)
            scores, targets = base_filtered_scores(model)
            # (a) standard
            r_std = ranks_from_scores(scores, targets, exclude_ids=None)
            m_std = metrics_from_ranks(r_std)
            per_seed["std"].append(m_std)
            # (b) frequent-5 excluded (full 297, gold->inf)
            r_freq = ranks_from_scores(scores, targets, exclude_ids=top5_ids)
            m_freq = metrics_from_ranks(r_freq)
            per_seed["freq"].append(m_freq)
            # unaffected subset for (b): queries whose gold not in top5
            unaff_b = np.array([i for i in range(len(FWD_TEST)) if QUERY_TAIL[i] not in top5_ids])
            per_seed["freq_unaff"].append(metrics_from_ranks(r_freq, subset=unaff_b))
            per_seed["std_on_freq_unaff"].append(metrics_from_ranks(r_std, subset=unaff_b))
            # (c) random control: average over draws
            n_goldex_list = []
            for d_i, dset in enumerate(draws):
                r_rand = ranks_from_scores(scores, targets, exclude_ids=dset)
                per_seed["rand"][d_i].append(metrics_from_ranks(r_rand))
                unaff_c = np.array([i for i in range(len(FWD_TEST)) if QUERY_TAIL[i] not in dset])
                per_seed["rand_unaff"][d_i].append(metrics_from_ranks(r_rand, subset=unaff_c))
                per_seed["std_on_rand_unaff"][d_i].append(metrics_from_ranks(r_std, subset=unaff_c))
                n_goldex_list.append(int(sum(1 for i in range(len(FWD_TEST)) if QUERY_TAIL[i] in dset)))
            per_seed["rand_n_goldex"].append(np.mean(n_goldex_list))
            del model

        def agg(metric_dicts, key):
            vals = [m[key] for m in metric_dicts]
            return float(np.mean(vals)), float(np.std(vals))

        # --- condition (a) standard
        for key, label in [("MRR", "MRR"), ("H1", "Hits@1"), ("H10", "Hits@10")]:
            mu, sd = agg(per_seed["std"], key)
            rows.append(dict(model=model_name, condition="(a) standard", view="all_297",
                             metric=label, mean=mu, std=sd, n=297, n_gold_excluded=0,
                             n_draws=1, note="no exclusion"))
        # --- condition (b) full 297 (gold->inf)
        n_gold_b = sum(1 for i in range(len(FWD_TEST)) if QUERY_TAIL[i] in top5_ids)
        for key, label in [("MRR", "MRR"), ("H1", "Hits@1"), ("H10", "Hits@10")]:
            mu, sd = agg(per_seed["freq"], key)
            rows.append(dict(model=model_name, condition="(b) freq-adjusted", view="full_297_goldinf",
                             metric=label, mean=mu, std=sd, n=297, n_gold_excluded=n_gold_b,
                             n_draws=1, note="exclude TOP5; gold-in-excluded=>rank inf"))
        # --- condition (b) unaffected subset (isolates distractor inflation)
        for key, label in [("MRR", "MRR"), ("H1", "Hits@1"), ("H10", "Hits@10")]:
            mu_b, sd_b = agg(per_seed["freq_unaff"], key)
            mu_a, sd_a = agg(per_seed["std_on_freq_unaff"], key)
            rows.append(dict(model=model_name, condition="(b) freq-adjusted", view="unaffected_subset",
                             metric=label, mean=mu_b, std=sd_b, n=297 - n_gold_b,
                             n_gold_excluded=n_gold_b, n_draws=1,
                             note=f"on queries w/ gold NOT in TOP5; paired std baseline={mu_a:.4f}"))
            rows.append(dict(model=model_name, condition="(a) std on (b)-unaffected", view="unaffected_subset",
                             metric=label, mean=mu_a, std=sd_a, n=297 - n_gold_b,
                             n_gold_excluded=0, n_draws=1, note="baseline for gain_b on same subset"))
        # --- condition (c) random control: average across draws then across seeds
        # For each seed, average metric across draws; then mean/std across seeds.
        def agg_rand(dct, key):
            per_seed_means = []
            for s_i in range(len(per_seed["std"])):
                draw_vals = [dct[d_i][s_i][key] for d_i in range(len(draws))]
                per_seed_means.append(np.mean(draw_vals))
            return float(np.mean(per_seed_means)), float(np.std(per_seed_means))
        n_gold_c = float(np.mean(per_seed["rand_n_goldex"])) if per_seed["rand_n_goldex"] else 0.0
        for key, label in [("MRR", "MRR"), ("H1", "Hits@1"), ("H10", "Hits@10")]:
            mu, sd = agg_rand(per_seed["rand"], key)
            rows.append(dict(model=model_name, condition="(c) random-control", view="full_297_goldinf",
                             metric=label, mean=mu, std=sd, n=297,
                             n_gold_excluded=round(n_gold_c, 3), n_draws=len(draws),
                             note="exclude 5 random non-top disease tails; gold-in=>inf; avg over draws"))
        for key, label in [("MRR", "MRR"), ("H1", "Hits@1"), ("H10", "Hits@10")]:
            mu_c, sd_c = agg_rand(per_seed["rand_unaff"], key)
            mu_ac, sd_ac = agg_rand(per_seed["std_on_rand_unaff"], key)
            rows.append(dict(model=model_name, condition="(c) random-control", view="unaffected_subset",
                             metric=label, mean=mu_c, std=sd_c, n=round(297 - n_gold_c, 2),
                             n_gold_excluded=round(n_gold_c, 3), n_draws=len(draws),
                             note=f"on queries w/ gold not in draw; paired std baseline={mu_ac:.4f}"))
            rows.append(dict(model=model_name, condition="(a) std on (c)-unaffected", view="unaffected_subset",
                             metric=label, mean=mu_ac, std=sd_ac, n=round(297 - n_gold_c, 2),
                             n_gold_excluded=0, n_draws=len(draws), note="baseline for gain_c on same subset"))

    fields = ["model", "condition", "view", "metric", "mean", "std", "n",
              "n_gold_excluded", "n_draws", "note"]
    with open(OUT_DIR / "distractor_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[A1] Wrote distractor_ablation.csv")
    return rows


# ===========================================================================
# ANALYSIS 3 (483/201 stratified ESM vs random delta-MRR)
# ===========================================================================
def analysis3():
    strata = {"esm": np.array([i for i in range(len(FWD_TEST)) if QUERY_STRATUM[i] == "esm"]),
              "random": np.array([i for i in range(len(FWD_TEST)) if QUERY_STRATUM[i] == "random"]),
              "all": np.array([i for i in range(len(FWD_TEST))])}
    # per-query 1/rank for each model/init/seed
    # store mrr_inv[model][init][seed] = array(297)
    inv = defaultdict(lambda: defaultdict(dict))
    for model_name, group in MODEL_GROUP.items():
        for init in ["esm2", "random"]:
            for seed in SEEDS:
                md = EXP / group / model_name / init / f"seed_{seed}"
                if not (md / "model.pt").exists():
                    print(f"  [A3] MISSING {md}")
                    continue
                model = load_model_dir(md)
                scores, targets = base_filtered_scores(model)
                r = ranks_from_scores(scores, targets, exclude_ids=None)
                inv[model_name][init][seed] = 1.0 / r
                del model
        print(f"  [A3] done {model_name}")

    rows = []
    # paired (by seed) delta computation, per model and pooled-across-models
    for model_name in list(MODEL_GROUP.keys()) + ["__POOLED__"]:
        for stratum_name, idx in strata.items():
            # gather per-seed MRR for esm and random
            if model_name == "__POOLED__":
                model_list = list(MODEL_GROUP.keys())
            else:
                model_list = [model_name]
            esm_seed_vals, rand_seed_vals, delta_seed_vals = [], [], []
            for seed in SEEDS:
                esm_per_model, rand_per_model = [], []
                for mn in model_list:
                    if seed in inv[mn]["esm2"] and seed in inv[mn]["random"]:
                        esm_per_model.append(float(np.mean(inv[mn]["esm2"][seed][idx])))
                        rand_per_model.append(float(np.mean(inv[mn]["random"][seed][idx])))
                if not esm_per_model:
                    continue
                esm_v = float(np.mean(esm_per_model))
                rand_v = float(np.mean(rand_per_model))
                esm_seed_vals.append(esm_v)
                rand_seed_vals.append(rand_v)
                delta_seed_vals.append(esm_v - rand_v)
            if not delta_seed_vals:
                continue
            esm_seed_vals = np.array(esm_seed_vals)
            rand_seed_vals = np.array(rand_seed_vals)
            delta_seed_vals = np.array(delta_seed_vals)
            # paired t-test across seeds
            n = len(delta_seed_vals)
            mean_d = float(np.mean(delta_seed_vals))
            sd_d = float(np.std(delta_seed_vals, ddof=1)) if n > 1 else float("nan")
            se = sd_d / math.sqrt(n) if n > 1 else float("nan")
            tstat = mean_d / se if (n > 1 and se > 0) else float("nan")
            try:
                from scipy import stats
                pval = float(stats.t.sf(abs(tstat), df=n - 1) * 2) if n > 1 else float("nan")
            except Exception:
                pval = float("nan")
            rows.append(dict(
                model=model_name, stratum=stratum_name, n_queries=int(len(idx)),
                n_seeds=n,
                mrr_esm2_mean=round(float(np.mean(esm_seed_vals)), 6),
                mrr_random_mean=round(float(np.mean(rand_seed_vals)), 6),
                delta_mrr=round(mean_d, 6),
                delta_mrr_sd_acrossseeds=round(sd_d, 6) if not math.isnan(sd_d) else "",
                paired_t=round(tstat, 4) if not math.isnan(tstat) else "",
                paired_p=round(pval, 5) if not math.isnan(pval) else "",
            ))
    fields = ["model", "stratum", "n_queries", "n_seeds", "mrr_esm2_mean",
              "mrr_random_mean", "delta_mrr", "delta_mrr_sd_acrossseeds",
              "paired_t", "paired_p"]
    with open(OUT_DIR / "esm_stratified.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("[A3] Wrote esm_stratified.csv")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "a1", "a2", "a3", "validate"])
    args = ap.parse_args()

    if args.only == "validate":
        # sanity: reproduce RotE esm2 seed_0 standard tail MRR (~0.3115)
        md = EXP / "runs_g0" / "RotE" / "esm2" / "seed_0"
        model = load_model_dir(md)
        scores, targets = base_filtered_scores(model)
        r = ranks_from_scores(scores, targets, exclude_ids=None)
        m = metrics_from_ranks(r)
        print("VALIDATE RotE esm2 seed0 standard:", m)
        print("  (metrics.json test tail_mrr=0.31150, hits@1=0.19529, hits@10=0.57912)")
        sys.exit(0)

    if args.only in ("all", "a2"):
        analysis2()
    if args.only in ("all", "a1"):
        analysis1()
    if args.only in ("all", "a3"):
        analysis3()
    print("DONE")
