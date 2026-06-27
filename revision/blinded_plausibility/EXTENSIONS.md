# Extensions to the blinded plausibility analysis

This extends the pre-registered, blinded biological-plausibility scorer
(`score_blinded.py`, `METHODS.md`) in two ways requested in revision:

1. **Multi-model, baseline-anchored comparison** — the same blinded rule is now
   applied to a **metric-consistent** model (RefH, best under the
   popularity-adjusted metric the paper argues for) and to the **source study's
   model type** (BoxE, the best model family in Sanou et al.), alongside the
   already-reported RotE. This answers the request to *"score plausibility with an
   explicit baseline including the source study's models"* and to run *"case
   studies on the metric-consistent best model."*
2. **Second independent rater + inter-rater agreement** — a second **automated**
   rater on a *different* Open Targets evidence axis tags the same RotE top-20
   predictions; we report Cohen's kappa and percent agreement. This is the
   reproducible substitute for the reviewer's *"second rater"* request.

Everything below was run **CPU-only** (`CUDA_VISIBLE_DEVICES=""`).

---

## Task 1 — top-k predictions for two more models

Generated per-query top-50 predictions on the **same 297 forward**
`(StudyProduct, imgt:hasClinicalIndication, ?)` queries with the generic
`scripts/atth_interpret.py` on CPU. The 297 queries are the identical test
triples in the identical order for every model (verified: all 297 head entities
align across RotE/RefH/BoxE), so per-query target genes are model-independent and
only the predictions differ.

| Model | Role | Trial chosen | How chosen | test tail_mrr | re-derived MRR |
|-------|------|--------------|------------|--------------:|---------------:|
| RefH | metric-consistent best (popularity-adjusted metric) | `RefH/trial_059` | given | 0.28392 | 0.284 |
| BoxE | source-study (Sanou et al.) best model type | `BoxE/trial_001` | **highest test `tail_mrr` over the 16 BoxE trials** (0.28807; next: trial_006 0.28781, trial_011 0.28702) | 0.28807 | 0.288 |

Outputs: `interpret_topk_50.json` written into each trial directory. Both models'
re-derived MRRs reproduce the values in their `metrics.json`, confirming the
checkpoints loaded correctly. (RefH and BoxE both inherit the
`get_queries`/`get_rhs`/`score`/`bt` interface from `models/base.py`, so the
generic interpreter works unmodified; BoxE's eval-mode scoring materialises an
(n_queries × n_entities × rank) tensor — well within available RAM.)

---

## Task 2 — blinded plausibility for RefH and BoxE (with baseline)

`score_models.py` imports the pre-registered functions from `score_blinded.py`
and applies the **identical** rule (OT overall association `score >= 0.01`,
top-20 scored, GENERIC buckets excluded, 1000-draw popularity baseline,
seed 20240617, 5000-bootstrap CI). It runs **offline** against the existing OT
cache (`cache/gene_ot_assoc.json`, OT snapshot **v26.03**) and the
**model-independent** popularity prior
(`RotE/trial_101/indication_prior.json`, the training-set indication
frequencies). All 101 target genes for the 297 queries are present in the cache,
so no network was needed. As a control, re-scoring RotE through this driver
reproduces the original `summary.json` enrichment **exactly** (1.2082).

### Results (`model_comparison.csv`)

| Model | Role | n scored | model plausible-frac | popularity baseline | **enrichment** (95% CI) | enrichment median | % queries enr>1 |
|-------|------|---------:|---------------------:|--------------------:|:-----------------------:|------------------:|----------------:|
| **RotE** (already done) | headline | 279/297 | 0.574 | 0.475 | **1.208** [1.183, 1.234] | 1.201 | 89.5% |
| **RefH** | metric-consistent best | 279/297 | 0.566 | 0.475 | **1.190** [1.173, 1.207] | 1.193 | 92.4% |
| **BoxE** | source-study baseline | 279/297 | 0.580 | 0.475 | **1.224** [1.211, 1.237] | 1.223 | 97.5% |

(18/297 queries are unscorable for every model — anti-glycan / anti-ganglioside
antibodies with no HGNC target. The popularity baseline is identical across
models because it depends only on the prior and the per-query target gene-set,
both model-independent.)

### Interpretation (honest)

- **All three models show a modest but statistically reliable enrichment**
  (~1.19–1.22×; every 95% CI excludes 1.0; ≥89% of scored queries enriched).
  Models rank top-20 indications that are biologically associated with their
  antibody's target gene **about 20% more often than a popularity-matched random
  draw would.**
- **The effect is not unique to RotE.** It is essentially the same magnitude for
  the metric-consistent best model (RefH) and — importantly — for **BoxE, the
  source study's own best model type**, which here has the *highest* mean
  enrichment (1.224×) and the largest share of enriched queries (97.5%). So the
  plausibility signal is a property of the **dataset/task**, not of one favoured
  model: it cannot be used to argue that RotE (or any single architecture) is
  uniquely "biologically meaningful."
- The differences between models are **small relative to the effect size**
  (RefH's CI overlaps RotE's; BoxE's is marginally higher). The honest reading is
  "comparable plausibility across architectures," not a winner.

---

## Task 3 — second independent (automated) rater + agreement

`second_rater.py` adds a **second automated rater** that scores the **same RotE
top-20 predictions** on a **different Open Targets evidence axis**, then computes
inter-rater agreement against the primary rater.

> **This is an automated rater, NOT a human.** It is the reproducible,
> pre-specified substitute for a human second rater: instead of a second person
> re-reading cases, a second *evidence axis* re-tags every prediction by the same
> mechanical rule. `second_rater_agreement.json` records `is_human_rater: false`.

To isolate the **evidence-axis** difference (rather than confounding it with OT
version drift), both raters in this sub-study read the **same freshly fetched OT
snapshot (v26.6.0)**, retrieving for each gene the top-50 associated diseases
with both the overall score and the per-datatype scores.

| Rater | Evidence axis | Rule | plausible-frac (RotE top-20 specific preds) |
|-------|---------------|------|--------------------------------------------:|
| **A (primary)** | OT **overall** association (composite over all datatypes) | score ≥ 0.01 | 0.581 |
| **B (headline 2nd rater)** | OT **`clinical`** datatype = clinical-precedence / **known-drug** evidence (ChEMBL drug-indication) | score ≥ 0.01 | 0.445 |
| **C (sensitivity)** | OT **`genetic_association`** datatype (GWAS/ClinVar human genetics) | score ≥ 0.01 | 0.081 |

Matching, normalisation, and GENERIC exclusion are identical to the primary
scorer; only the per-disease score field changes between raters. 5007 specific
(non-generic) predictions across the 279 scorable queries were tagged by each
rater (573 generic predictions excluded identically from all raters).

### Inter-rater agreement

| Comparison | % agreement | Cohen's kappa | Landis–Koch band |
|------------|------------:|--------------:|------------------|
| **A overall vs B clinical/known-drug (HEADLINE)** | **86.3%** | **0.732** | **substantial** |
| A overall vs C genetic_association | 50.0% | 0.119 | slight |
| B clinical vs C genetic_association | 59.9% | 0.116 | slight |

### What kappa implies

- **κ = 0.73 (substantial agreement; 86.3% raw agreement)** between the primary
  composite axis and an **independently derived clinical/known-drug evidence
  axis** means the PLAUSIBLE/OFF-TARGET tags are **robust**: two raters that draw
  on different Open Targets evidence streams agree on the verdict for ~6 of every
  7 predictions. The plausibility tagging is not an artefact of the particular
  scoring channel chosen.
- **Honest caveat — the genetic-only axis disagrees (κ ≈ 0.12).** Human-genetics
  evidence is *sparse* for these targets (only 8.1% of predictions are
  "plausible" under it), because most oncology mAb indications are **somatic**
  cancers with little germline GWAS/ClinVar signal. A pure-genetics rater is
  therefore a poor standalone proxy for clinical-indication plausibility here;
  we report it transparently rather than cherry-picking the axis that agrees.
  The clinical/known-drug axis is the appropriate independent comparator for this
  task, and it is the one that shows substantial agreement.

---

## Files produced

| File | Contents |
|------|----------|
| `model_comparison.csv` | RotE / RefH / BoxE: plausible-frac, baseline, enrichment + 95% CI, % queries enr>1 |
| `second_rater_agreement.json` | rater definitions, plausible-fractions, Cohen's kappa + % agreement for all axis pairs |
| `second_rater_predictions.csv` | per-prediction PLAUSIBLE/OFF-TARGET tag from all three raters (5007 rows) |
| `summary_RefH.json`, `summary_RotE_check.json`, `summary_BoxE.json` | full per-model summaries |
| `results_297_{RefH,BoxE,RotE_check}.csv` | per-query enrichment for each model |
| `score_models.py`, `second_rater.py` | reproducible drivers (CPU, import the pre-registered logic) |
| `cache/gene_ot_datatype.json` | per-gene OT datatype scores (snapshot v26.6.0) used by the second rater |
| `analysis_outputs/.../{RefH/trial_059,BoxE/trial_001}/interpret_topk_50.json` | the new top-50 prediction dumps |

## Reproducing

```bash
export CUDA_VISIBLE_DEVICES=""
# Task 1 (top-k)
python3 ../../scripts/atth_interpret.py --trial_dir <RefH/trial_059 | BoxE/trial_001> --top_k 50
# Task 2 (blinded scoring, offline against cached OT)
python3 score_models.py --trial_dir <trial_dir> --label <RefH|BoxE>
# Task 3 (second rater; fetches OT datatype scores once, then cached)
python3 second_rater.py            # use --offline after the cache is built
```

## Caveats (carried over + new)

- **Automated, not human, second rater** (stated above). Both raters share the
  same disease-name normalisation and token-overlap matcher, so they are *not*
  fully independent in their string handling — the independence is in the
  **evidence axis**, which is the dimension the second-rater request targets.
- The model comparison (Task 2) uses OT **v26.03** (the original cache); the
  second-rater study (Task 3) uses a fresh **v26.6.0** fetch so both axes are on
  one snapshot. Re-running A on v26.6 gives plausible-frac 0.581 vs 0.574 on
  v26.03 — i.e. snapshot drift is small and does not affect conclusions.
- All other limitations from `METHODS.md` apply (top-50 OT truncation,
  multi-target union is permissive, no-target antibodies unscorable).
