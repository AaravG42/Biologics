"""Aggregate multi-seed results -> mean +/- 95% CI + significance tests.

Two run sources are kept SEPARATE to avoid key collisions:
  - main campaign (runs_g0/g1/g3): inits 'esm2' (additive ESM scheme) and 'random'.
  - encoder ablation (encoder_ablation_runs): inits 'esm2'(PCA), 'ablang', 'igbert',
    all magnitude-matched; here 'esm2' means the PCA-pipeline ESM (NOT the additive one).

Outputs: benchmark_multiseed.csv (#2), esm_vs_random.csv (#1a),
encoder_ablation.csv (#1b), aggregate_summary.json.
"""
from __future__ import annotations
import csv, glob, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

ROOT = Path(__file__).parent
MAIN_DIRS = ["runs_g0", "runs_g1", "runs_g3"]
ENC_DIRS = ["encoder_ablation_runs"]
METRIC_KEYS = ["tail_mrr", "tail_mr", "tail_hits@1", "tail_hits@3", "tail_hits@10"]


def collect(dirs):
    rows = defaultdict(dict)  # (model, init) -> {seed: testmetrics}
    for rd in dirs:
        for mp in glob.glob(str(ROOT / rd / "**" / "metrics.json"), recursive=True):
            d = json.loads(Path(mp).read_text())
            rows[(d["model"], d["init"])][int(d["seed"])] = d["metrics"]["test"]
    return rows


def arr(rows, key, init):
    out = {}
    for (model, ini), seedmap in rows.items():
        if ini != init:
            continue
        seeds = sorted(seedmap)
        out[model] = (seeds, np.array([seedmap[s][key] for s in seeds], float))
    return out


def mean_ci(x):
    x = np.asarray(x, float); n = len(x)
    m = float(np.mean(x)); sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return m, sd, (m - half, m + half), n


def paired(a_map, b_map):
    """paired across common seeds. a_map/b_map = (seeds, values)."""
    sa, a = a_map; sb, b = b_map
    common = sorted(set(sa) & set(sb))
    if len(common) < 2:
        return None
    av = np.array([a[sa.index(s)] for s in common]); bv = np.array([b[sb.index(s)] for s in common])
    diff = float(np.mean(av - bv)); tp = wp = None
    if HAVE_SCIPY:
        try: tp = float(sps.ttest_rel(av, bv).pvalue)
        except Exception: pass
        try: wp = float(sps.wilcoxon(av, bv).pvalue) if np.any(av != bv) else 1.0
        except Exception: pass
    return {"mean_diff": diff, "t_p": tp, "wilcoxon_p": wp, "n": len(common)}


def main():
    main_rows = collect(MAIN_DIRS)
    enc_rows = collect(ENC_DIRS)
    summary = {"n_main_runs": sum(len(v) for v in main_rows.values()),
               "n_enc_runs": sum(len(v) for v in enc_rows.values()), "have_scipy": HAVE_SCIPY}
    print(f"main runs: {summary['n_main_runs']}, encoder-ablation runs: {summary['n_enc_runs']}\n")

    # ---- #2 multi-seed benchmark (additive ESM-2) ----
    esm = {k: arr(main_rows, k, "esm2") for k in METRIC_KEYS}
    models = sorted(esm["tail_mrr"], key=lambda m: -np.mean(esm["tail_mrr"][m][1]))
    bench = []
    print("== #2 Multi-seed benchmark (ESM-2 additive), by mean MRR ==")
    with (ROOT / "benchmark_multiseed.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "n", "MRR_mean", "MRR_std", "MR_mean", "MR_std", "H@1_mean", "H@10_mean"])
        for model in models:
            mr, sd, ci, n = mean_ci(esm["tail_mrr"][model][1])
            MR = float(np.mean(esm["tail_mr"][model][1])); MRsd = float(np.std(esm["tail_mr"][model][1], ddof=1))
            h1 = float(np.mean(esm["tail_hits@1"][model][1])); h10 = float(np.mean(esm["tail_hits@10"][model][1]))
            bench.append({"model": model, "mrr_mean": mr, "mrr_std": sd, "mr_mean": MR, "h1": h1, "h10": h10, "n": n})
            w.writerow([model, n, f"{mr:.4f}", f"{sd:.4f}", f"{MR:.2f}", f"{MRsd:.2f}", f"{h1*100:.1f}", f"{h10*100:.1f}"])
            print(f"  {model:7} MRR {mr:.4f}+/-{sd:.4f}  MR {MR:.1f}  H@1 {h1*100:.1f}%  H@10 {h10*100:.1f}%")
    summary["benchmark"] = bench

    top = models[:3]
    print("\n  pairwise MRR (additive ESM-2) among top-3:")
    summary["top3_pairwise"] = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            r = paired(esm["tail_mrr"][top[i]], esm["tail_mrr"][top[j]])
            summary["top3_pairwise"].append({"a": top[i], "b": top[j], **(r or {})})
            print(f"    {top[i]} vs {top[j]}: dMRR={r['mean_diff']:+.4f}  t_p={r['t_p']}  wilcoxon_p={r['wilcoxon_p']}")

    # ---- #1a ESM-2 vs random (main campaign) ----
    em = arr(main_rows, "tail_mrr", "esm2"); rm = arr(main_rows, "tail_mrr", "random")
    print("\n== #1a ESM-2(additive) vs random (clean data), dMRR = esm - random ==")
    ab = []
    with (ROOT / "esm_vs_random.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "n", "esm_MRR", "random_MRR", "dMRR", "t_p", "wilcoxon_p"])
        for model in sorted(set(em) & set(rm)):
            r = paired(em[model], rm[model])
            if not r: continue
            e = float(np.mean(em[model][1])); rr = float(np.mean(rm[model][1]))
            ab.append({"model": model, "esm": e, "random": rr, **r})
            w.writerow([model, r["n"], f"{e:.4f}", f"{rr:.4f}", f"{r['mean_diff']:+.4f}", r["t_p"], r["wilcoxon_p"]])
            star = " *" if (r["t_p"] is not None and r["t_p"] < 0.05) else ""
            print(f"  {model:7} esm {e:.4f} vs rand {rr:.4f}  dMRR={r['mean_diff']:+.4f}  t_p={r['t_p']}{star}")
    summary["esm_vs_random"] = ab

    # ---- #1b encoder ablation per model (PCA pipeline; baseline = main random) ----
    print("\n== #1b Encoder ablation (PCA pipeline, magnitude-matched), MRR ==")
    enc_out = {}
    with (ROOT / "encoder_ablation.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "init", "n", "MRR_mean", "MRR_std", "dMRR_vs_random", "t_p"])
        enc_models = sorted({m for (m, _i) in enc_rows})
        for model in enc_models:
            rand_map = rm.get(model)  # random baseline from main campaign
            print(f"  [{model}]  (random baseline {np.mean(rand_map[1]):.4f})" if rand_map else f"  [{model}]")
            enc_out[model] = []
            for ini in ["esm2", "ablang", "igbert"]:
                d = arr(enc_rows, "tail_mrr", ini)
                if model not in d: continue
                m_, sd_, ci_, n_ = mean_ci(d[model][1])
                rcmp = paired(d[model], rand_map) if rand_map else None
                dv = rcmp["mean_diff"] if rcmp else float("nan"); tp = rcmp["t_p"] if rcmp else None
                enc_out[model].append({"init": ini, "mrr_mean": m_, "mrr_std": sd_, "n": n_,
                                       "dMRR_vs_random": dv, "t_p": tp})
                w.writerow([model, ini, n_, f"{m_:.4f}", f"{sd_:.4f}", f"{dv:+.4f}", tp])
                star = " *" if (tp is not None and tp < 0.05) else ""
                print(f"    {ini:7} MRR {m_:.4f}+/-{sd_:.4f}  dvs.random {dv:+.4f}  t_p={tp}{star}")
    summary["encoder_ablation"] = enc_out

    (ROOT / "aggregate_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWROTE aggregate_summary.json + CSVs to {ROOT}")


if __name__ == "__main__":
    main()
