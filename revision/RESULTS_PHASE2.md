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

## 6. Transductive controlled split (final — 9 models × 5 seeds, 45/45)

To allow a like-for-like comparison with Sanou et al.'s *transductive* KGE setting
(and an honest contrast with our inductive directional split), we built a
transductive controlled split (`scripts/create_imgt_transductive_clinical.py`):
all 9,611 entities appear in train, reciprocal leakage = 0, forward-clinical
triples split at the triple level (train 27,622 / valid 181 / test 180; OOV = 0).
Multi-seed benchmark (additive ESM-2, 5 seeds each; `experiments/transductive_benchmark.csv`):

| Model | transductive MRR (mean±std) | MR | Hits@1 | Hits@10 | (inductive MRR) |
|---|---|---|---|---|---|
| TransE | 0.2894 ± 0.0046 | 36.1 | 15.7% | 61.7% | 0.3131 |
| RefE | 0.2808 ± 0.0112 | 34.2 | 15.4% | 57.6% | 0.3113 |
| RotE | 0.2760 ± 0.0160 | 35.0 | 15.2% | 56.3% | 0.3116 |
| RotH | 0.2667 ± 0.0029 | 35.9 | 13.3% | 58.0% | 0.2632 |
| AttH | 0.2497 ± 0.0038 | 84.0 | 14.4% | 48.0% | 0.2884 |
| MurE | 0.2444 ± 0.0061 | 39.4 | 14.7% | 45.9% | 0.2943 |
| AttE | 0.2434 ± 0.0038 | 39.4 | 13.9% | 48.3% | 0.2954 |
| CP | 0.2354 ± 0.0018 | 48.9 | 14.0% | 46.2% | 0.2904 |
| RefH | 0.2194 ± 0.0032 | 1176.7 | 13.9% | 39.6% | 0.2844 |

**TransE is again the best model** in the transductive setting; the Euclidean
models (TransE/RefE/RotE) again lead. Absolute MRRs are lower than on the inductive
split, but the two are not directly comparable (different test sets — the
transductive test is the Sanou-comparable one). The model ranking is stable across
both settings, reinforcing that the single-seed "RotE best" selection was not robust.

## 7. Popularity-adjusted metrics are largely an artifact (distractor-removal ablation)

Addressing Referee 2 Moderate concern 2. On the 297 clinical test queries
(`experiments/ablations/`): excluding the 5 most-frequent training tail nodes
inflates MRR by **+0.089 (RotE)** / **+0.083 (TransE)**, but excluding 5 *random*
non-frequent disease tails (≥20 draws) inflates it by only **+0.0009** — a
~100× difference. Moreover, **117/297 (39.4%)** test queries have one of those
five nodes *as their ground-truth tail* (Solid tumours 53, NSCLC 22, Cancers 16,
MM 15, NHL 11), so honestly scored over all 297, deleting them *collapses* MRR
from 0.31 → 0.12. **Conclusion:** the popularity-adjusted metric mostly removes
common distractors (and legitimate answers); it must be demoted from any headline
claim. Standard MRR (with CIs) is the honest primary metric.

## 8. The ESM-2 benefit is diffuse, not localized (483/201 stratification)

Stratifying the 297 queries by whether the query's mAb received ESM init: ESM-vs-
random ΔMRR is **+0.0066 (p=0.016)** on the 229 ESM-mAb queries vs **+0.0119
(p=0.0009)** on the 68 random-mAb queries. The benefit does **not** concentrate on
ESM-initialised mAbs — it propagates through the shared graph geometry. So the
"201 random mAbs dilute the effect" framing does not hold; the effect is real but
systemic. (`experiments/ablations/esm_stratified.csv`.)

## 9. Plausibility is model-agnostic and robust (baselines + second rater)

Same pre-registered, blinded rule applied to three models (`blinded_plausibility/`):

| Model | Enrichment (95% CI) | % queries >1 |
|---|---|---|
| RotE (headline) | 1.21× [1.18, 1.23] | 89.5% |
| RefH (metric-consistent best) | 1.19× [1.17, 1.21] | 92.4% |
| BoxE (Sanou et al.'s model — baseline) | 1.22× [1.21, 1.24] | 97.5% |

The signal is a property of the **task, not of RotE** (BoxE, the source study's own
model, is highest), which supplies the requested source-study baseline and the
metric-consistent case-study model. A second **independent automated rater** (Open
Targets clinical/known-drug axis vs the overall axis) gives Cohen's **κ = 0.73**
(86.3% agreement) → the tags are robust across evidence streams. (The second rater
is automated, not human — stated explicitly.)

## 10. External held-out validation (no temporal data available)

The KG has **no timestamps** (confirmed) and **anonymised antibodies** (no INN), so
a strict temporal split and ClinicalTrials.gov-by-name are both infeasible — stated
plainly. As the defensible substitute we used Open Targets target-level
clinical-trial evidence (101 target genes; accessed 2026-06-27) for mAb–indication
pairs **not in training** (`external_validation/`):

- **vs random:** strong — corroborated held-out pairs (n=99) recall@10 **0.566**,
  median rank **8/247** (~14× over random), recovering non-obvious hypotheses the
  popularity prior misses (EPCAM→colon rank 1 vs 42; CD38→Merkel rank 5 vs 58;
  ERBB2→biliary rank 8 vs 95).
- **vs the popularity prior:** roughly **at parity in aggregate** (broad set
  enrichment 0.95×; CIs include 1.0), because the external trial space is dominated
  by common cancers the prior already ranks high.

This is *not* a prospective/temporal validation (impossible here); it shows real
signal over random with honest parity-vs-prior in aggregate.

## Bottom line for the revision

1. Leakage audit remains the headline contribution.
2. Honest benchmark: top models tied at MRR ≈ 0.31 (modest, post-leakage).
3. ESM-2 gives a small, significant, *genuine* gain on the best models — with the
   magnitude mechanism characterised (anticipating the obvious confound).
4. Antibody-specific encoders add nothing here.
5. Systematic blinded plausibility is 1.21×, not 1.8×.
6. Popularity-adjusted metrics are a distractor-removal artifact (≈100× the gain of
   removing random nodes; remove ground truth for 39% of queries) — demote them;
   standard MRR with CIs is the honest primary metric.
7. The plausibility signal is model-agnostic (RotE 1.21× / RefH 1.19× / BoxE 1.22×,
   all CIs > 1) and robust across raters (κ = 0.73). The ESM benefit is diffuse,
   not localized to ESM-initialised mAbs.
8. Transductive split (final, 5 seeds): TransE again best (0.289); model ranking
   stable vs the inductive split. External validation: real signal vs random, at
   parity vs the popularity prior in aggregate; a strict temporal validation is
   impossible (no timestamps).

## Reproducibility

- `experiments/run_multiseed.py` — #1 (benchmark + ESM-vs-random).
- `experiments/run_encoder_ablation.py` — #4 (ESM/AbLang/IgBert, matched).
- `experiments/run_magnitude_control.py` — #3 (scaled-random control).
- `experiments/aggregate_results.py` — tables + significance.
- `blinded_plausibility/score_blinded.py` — #5.
- Per-trial seed inside `train_single_trial`; init/control seeds fixed (42/123).
- Open Targets snapshot v26.03 (cached); see METHODS files.
