# Phase-2 Results: Clean-data multi-seed benchmark, ESM-2 ablation, blinded validation

All experiments are on the **leakage-free directional** dataset
(`IMGT_no_leakage_directional_clinical`, 297 forward clinical-indication test
triples). Training reuses the exact paper pipeline (`train_single_trial`); each
condition is run over **5 seeds** (0–4). Scripts and per-run metrics are in
`revision/experiments/`; aggregation in `aggregate_results.py`.

These results are intended to replace single-seed / hand-selected claims in the
submitted manuscript with repeated-run, significance-tested, and magnitude-
controlled evidence.

## 1. Multi-seed benchmark (additive ESM-2 init, 5 seeds)

| Model | Geometry | MRR (mean±std) | MR | Hits@1 | Hits@10 |
|---|---|---|---|---|---|
| TransE | Euclidean | 0.3131 ± 0.0022 | 32.0 | 19.0% | 57.8% |
| RotE | Euclidean | 0.3116 ± 0.0038 | 23.7 | 18.9% | 59.3% |
| RefE | Euclidean | 0.3113 ± 0.0068 | 23.7 | 18.8% | 58.0% |
| AttE | Euclidean | 0.2954 ± 0.0060 | 23.7 | 18.8% | 54.0% |
| MurE | Euclidean | 0.2943 ± 0.0027 | 24.7 | 18.2% | 54.6% |
| CP | Euclidean | 0.2904 ± 0.0240 | 26.3 | 18.5% | 53.9% |
| AttH | Hyperbolic | 0.2884 ± 0.0067 | 66.5 | 18.8% | 49.4% |
| RefH | Hyperbolic | 0.2844 ± 0.0032 | 427.8 | 17.8% | 52.5% |
| RotH | Hyperbolic | 0.2632 ± 0.0076 | 34.2 | 14.7% | 49.5% |

**The top three (TransE, RotE, RefE) are statistically indistinguishable on MRR**
— pairwise paired t-tests: TransE–RotE p=0.38, TransE–RefE p=0.58, RotE–RefE
p=0.93. The manuscript's single-seed selection of RotE (0.316) is *within* the
multi-seed range, but it is not significantly the best. Model selection among
TransE/RotE/RefE should be pre-registered and justified (note the
popularity-adjusted metric in the submission favours RefH/MurE, so the choice
matters). Euclidean models clearly beat hyperbolic here.

## 2. ESM-2 initialization on clean data (5 seeds, paired t-test)

additive ESM-2 vs uniform-random init:

| Model | ΔMRR (ESM − random) | p | |
|---|---|---|---|
| TransE | +0.0169 | **0.0002** | significant |
| RefE | +0.0139 | **0.0008** | significant |
| RotE | +0.0114 | **0.0029** | significant |
| CP | +0.0178 | 0.23 | n.s. (high variance) |
| AttH | +0.0055 | 0.15 | n.s. |
| AttE | +0.0022 | 0.27 | n.s. |
| MurE | +0.0019 | 0.08 | n.s. |
| RotH | +0.0011 | 0.91 | n.s. |
| RefH | −0.0000 | 0.40 | n.s. |

**ESM-2 initialization gives a small but statistically significant improvement on
the three best models** (TransE/RefE/RotE) on the clean dataset — directly
addressing the criticism that the ESM-2 benefit was shown only on leaky data,
with a single number and no variance/significance.

## 3. Is the gain sequence information or init magnitude? (decisive control)

The additive-ESM scheme gives ESM-initialised mAb entities a row-norm of ~11 vs
~0.15 for all other (random) entities — a **~76× magnitude gap**. We disentangle
magnitude from sequence-direction with a 2×2 (TransE; RefE consistent):

| Condition | init magnitude | mAb direction | TransE MRR |
|---|---|---|---|
| additive-ESM | large (~11) | **ESM** | **0.3131 ± 0.0020** |
| scaled-random | large (~11) | random | 0.2993 ± 0.0012 |
| ESM-pca (matched) | small (~0.15) | ESM | 0.2976 ± 0.0039 |
| uniform-random | small (~0.15) | random | 0.2962 ± 0.0017 |

(RefE: additive-ESM 0.3113 ± 0.0060 vs scaled-random 0.2983 ± 0.0030.)

**Interpretation:** the gain is *genuinely* from the ESM sequence directions — at
identical large magnitude, ESM directions beat random directions (+0.014). But
the effect is **contingent on large initialisation magnitude**: neither
magnitude alone (scaled-random ≈ uniform-random) nor sequence directions at
matched small scale (ESM-pca ≈ uniform-random) help. This is a previously
unreported confound in this setting; the honest claim is "ESM-2 sequence
initialisation helps, conditional on init scale."

## 4. Antibody-specific encoders (AbLang, IgBert)

Under a fair, magnitude-matched pipeline (per-mAb concat of heavy/light
mean-pooled embeddings → PCA-640 → L2-norm → matched scale), 5 seeds:

| Model | random | ESM-2 | AbLang | IgBert |
|---|---|---|---|---|
| RotE | 0.3002 | 0.2991 | 0.3005 | 0.3011 |
| TransE | 0.2962 | 0.2976 | 0.2960 | 0.2965 |

All differences vs random are non-significant (p>0.38). **Antibody-specific
encoders provide no advantage over ESM-2** here, and at matched magnitude no
sequence encoder beats random — consistent with §3. (Caveat: AbLang sees only
the ~157-residue Fv region; see `experiments/encoder_inits/METHODS.md`.)

## 5. Systematic, blinded pharmacological plausibility (all 297 queries)

Pre-registered Open Targets threshold (≥0.01), blinded to ground truth, across
all queries (full method in `revision/blinded_plausibility/METHODS.md`):

- **279/297 queries scorable** (18 antibodies target non-protein antigens —
  glycans/gangliosides — and are reported as unscorable, not dropped).
- Model plausible-fraction **0.574** vs popularity baseline **0.475** →
  **enrichment 1.21× (95% CI 1.18–1.23)**; **89.5%** of queries enrichment > 1;
  stable across thresholds 0.01/0.05/0.1.
- Reproduces the 5 original case studies (blinded counts within +0 to +2).

The manuscript's headline **1.8× was over 5 hand-selected cases**; the
systematic, blinded, all-query value is **1.21×** — a real but modest signal that
removes the cherry-picking vulnerability.

## Bottom line for the revision

1. Leakage audit remains the headline contribution.
2. Honest benchmark: top models tied at MRR ≈ 0.31 (modest, post-leakage).
3. ESM-2 gives a small, significant, *genuine* gain on the best models — with the
   magnitude mechanism characterised (anticipating the obvious confound).
4. Antibody-specific encoders add nothing here.
5. Systematic blinded plausibility is 1.21×, not 1.8×.

## Reproducibility

- `experiments/run_multiseed.py` — #1 (benchmark + ESM-vs-random).
- `experiments/run_encoder_ablation.py` — #4 (ESM/AbLang/IgBert, matched).
- `experiments/run_magnitude_control.py` — #3 (scaled-random control).
- `experiments/aggregate_results.py` — tables + significance.
- `blinded_plausibility/score_blinded.py` — #5.
- Per-trial seed inside `train_single_trial`; init/control seeds fixed (42/123).
- Open Targets snapshot v26.03 (cached); see METHODS files.
