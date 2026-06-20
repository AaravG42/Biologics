# Revision analyses (post-review)

> **Status:** These analyses were produced *after* the initial submission, in
> response to the referee reports. They are **not part of the submitted
> manuscript** and, in several places, **update or supersede** claims in it.
> They support a planned, honest revision.

Full write-up with all tables: [`RESULTS_PHASE2.md`](RESULTS_PHASE2.md).

## What these add / correct, in brief

1. **Multi-seed benchmark (5 seeds).** TransE/RotE/RefE are **statistically tied**
   on MRR (~0.31; pairwise p>0.37) — the single-seed "RotE is best" selection is
   not supported. `experiments/run_multiseed.py`, `experiments/benchmark_multiseed.csv`.
2. **ESM-2 on clean data, with significance.** ESM-2 init gives a small but
   significant gain on the top models (TransE +0.017 p=0.0002; RefE +0.014;
   RotE +0.011). `experiments/esm_vs_random.csv`.
3. **Magnitude control (the key caveat).** The additive-ESM scheme makes mAb
   embeddings ~76× larger than other entities; a 2×2 control shows the gain is
   genuine sequence-direction information but **contingent on large init
   magnitude**. `experiments/run_magnitude_control.py`.
4. **Antibody-specific encoders.** AbLang/IgBert give **no advantage** over ESM-2
   under a magnitude-matched pipeline. `experiments/run_encoder_ablation.py`,
   `experiments/encoder_inits/`.
5. **Systematic, blinded pharmacological plausibility over all queries.** The
   manuscript's headline "1.8× enrichment" was over 5 hand-selected cases; a
   pre-registered, blinded score across all 279 scorable queries gives **1.21×**
   (95% CI 1.18–1.23). `blinded_plausibility/`.

## Layout

- `RESULTS_PHASE2.md` — consolidated results and interpretation.
- `experiments/` — training/eval/aggregation scripts and **per-run `metrics.json`
  records** for every (model, init, seed). Trained `model.pt` checkpoints are
  **omitted** (3+ GB) but are fully regenerable from the scripts and fixed seeds.
- `blinded_plausibility/` — the standalone blinded scorer, per-query results,
  cached Open Targets responses (snapshot v26.03), and methods.

## Reproduce

```bash
# from the repository root (see ../docs/REPRODUCE.md for environment)
python revision/experiments/run_multiseed.py --gpu 0
python revision/experiments/run_encoder_ablation.py --model TransE --gpu 0
python revision/experiments/run_magnitude_control.py --gpu 0
python revision/experiments/aggregate_results.py
python revision/blinded_plausibility/score_blinded.py --offline   # uses cached OT responses
```
