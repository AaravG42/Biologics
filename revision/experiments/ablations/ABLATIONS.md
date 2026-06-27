# Ablation analyses on already-trained IMGT clinical KG-embedding models

All computation runs on **CPU only** (`CUDA_VISIBLE_DEVICES=""`, repo CPU fallback).
Nothing outside `revision/experiments/ablations/` was modified.

- Dataset: `data/IMGT_no_leakage_directional_clinical` (inductive, directional clinical set).
- Forward clinical relation `imgt:hasClinicalIndication` = **relation id 6**.
- Evaluation target: the **297 forward clinical test triples** (tail prediction: `StudyProduct, hasClinicalIndication, ?disease`).
- Models loaded exactly as `clinical_eval.py` does, except the config is read from the
  embedded `metrics.json["config"]` block (these checkpoints ship the config inside
  `metrics.json`, not a separate `config.json`). The class is built from `models.<name>`,
  `model.pt` is loaded with `map_location="cpu"`, `model.eval()`.
- **Pipeline validation:** the standard (no-exclusion) tail metrics reproduce the stored
  `metrics.json` exactly for RotE/esm2/seed_0 — MRR 0.311504 (stored 0.311504),
  Hits@1 0.195286 (0.195286), Hits@10 0.579125 (0.579125). The ranking code mirrors
  `KGModel.get_ranking` (filtered setting: all known tails for `(head, rel)` plus the gold
  are masked to -1e6; rank = 1 + #candidates scoring >= gold).

Script: `run_ablations.py` (in this directory). Outputs: `distractor_ablation.csv`,
`excluded_as_gold.csv`, `esm_stratified.csv`.

The 5 frequent TRAIN tail nodes excluded under popularity adjustment, with their measured
forward-clinical train frequencies (confirmed to be the top 5):

| node | train fwd freq |
|---|---|
| imgt:Solid_tumors | 149 |
| imgt:Cancers_non-small_cell_lung_(NSCLC) | 48 |
| imgt:Cancers | 42 |
| imgt:Multiple_myeloma_(MM) | 39 |
| imgt:Non-Hodgkins_lymphoma_(NHL) | 39 |

---

## Analysis 1 — Distractor-removal ablation

**Question.** Does "popularity adjustment" (deleting the 5 most frequent train tail nodes
from the candidate pool before ranking) inflate the metric by removing *common distractors*,
or is the gain genuine? Test: compare the gain from removing the 5 *frequent* nodes (b) to the
gain from removing 5 *random non-frequent* disease tails (c).

**Models / checkpoints.** RotE and TransE, **esm2** init, **all 5 seeds** (mean ± std across
seeds reported).

**Conditions.**
- (a) standard — no exclusion.
- (b) popularity-adjusted — exclude the 5 frequent nodes above.
- (c) control — exclude 5 random disease tails sampled (without replacement) from the
  242 forward-clinical disease tails that are *not* in the top-5 set; **50 random draws**
  (fixed RNG seed 12345, same draws reused for both models and all seeds);
  reported as mean over draws then mean ± std over seeds.

**Handling of "excluded node is the gold answer".** When an excluded node is itself the
ground-truth tail of a query, that query's correct answer has been removed from the pool.
**Policy chosen: rank = infinity (counts as a miss; MRR contribution 0, not Hits@k).** This
keeps n = 297 fixed across all conditions so the views are directly comparable, and it is the
honest accounting of the cost of deleting legitimate answers. Queries affected:
**117 / 297 under (b)**; on average **4.12 / 297 per random draw under (c)**.

Because (b) deletes the gold for 117 queries, the all-297 number conflates two effects, so we
report **two views**:
1. **full_297_goldinf** — all 297 queries, gold-removed = rank inf. Shows the *net* effect.
2. **unaffected_subset** — only queries whose gold is NOT in the excluded set, with the
   condition-(a) baseline recomputed on that *same* subset (paired). This **isolates the pure
   distractor-removal effect** on the queries that still have a valid answer.

### Results (mean over 5 seeds)

**Net effect — all 297 queries, gold-removed scored as miss (MRR):**

| model | (a) standard | (b) freq-adjusted | (c) random-control |
|---|---|---|---|
| RotE   | 0.3116 | 0.1203 | 0.3112 |
| TransE | 0.3131 | 0.1260 | 0.3127 |

Net gain `(b)-(a)` ≈ **−0.19** (both models); net gain `(c)-(a)` ≈ **−0.0004**. I.e. once you
honestly score the removed answers, deleting the frequent nodes *destroys* overall MRR
(collapsing 0.31 → 0.12) because the deleted nodes are the gold answer for 39 % of queries,
whereas deleting random nodes does essentially nothing.

**Isolated distractor effect — unaffected subset, paired baseline (gain = adjusted − standard on the same queries):**

| model | metric | gain (b) freq-removal | gain (c) random-removal | ratio b/c |
|---|---|---|---|---|
| RotE   | MRR     | **+0.0894** (0.1090→0.1985) | +0.0009 (0.3148→0.3157) | ~99× |
| RotE   | Hits@1  | **+0.0678** (0.0122→0.0800) | +0.0002 | — |
| RotE   | Hits@10 | **+0.1178** (0.3611→0.4789) | +0.0025 | ~47× |
| TransE | MRR     | **+0.0833** (0.1246→0.2079) | +0.0010 (0.3161→0.3172) | ~80× |
| TransE | Hits@1  | **+0.0822** (0.0200→0.1022) | +0.0004 | — |
| TransE | Hits@10 | **+0.1067** (0.3511→0.4578) | +0.0026 | ~41× |

### Conclusion (Analysis 1)

**Popularity adjustment inflates the metric by removing common distractors.** On the surviving
(rare-disease) queries, deleting the 5 frequent nodes nearly *doubles* MRR (+0.083–0.089 on a
baseline of ~0.11–0.12) and lifts Hits@1 by +0.07–0.08, while deleting 5 random non-frequent
disease tails produces a negligible +0.001 — an ~80–100× difference. The improvement is
therefore an artifact of removing the hardest, most-confusable competitors (the frequent
oncology super-classes such as "Solid_tumors"), not a genuine increase in retrieval quality.
Worse, the adjustment deletes the *true* answer for 117/297 (39 %) of test queries; scored
honestly, the popularity-adjusted MRR is *lower* than the unadjusted one, not higher. The
popularity-adjusted number should not be reported as a fair measure of clinical retrieval
performance.

---

## Analysis 2 — Per-query excluded-node-as-ground-truth accounting

Across the 297 forward clinical test triples, **117 (39.4 %)** have one of the 5 excluded
frequent nodes as their ground-truth tail (see `excluded_as_gold.csv`):

| excluded tail | entity id | train fwd freq | all-splits fwd freq | # test queries with this gold |
|---|---|---|---|---|
| imgt:Solid_tumors | 6058 | 149 | 243 | 53 |
| imgt:Cancers_non-small_cell_lung_(NSCLC) | 1272 | 48 | 80 | 22 |
| imgt:Cancers | 1238 | 42 | 77 | 16 |
| imgt:Multiple_myeloma_(MM) | 4094 | 39 | 70 | 15 |
| imgt:Non-Hodgkins_lymphoma_(NHL) | 4129 | 39 | 59 | 11 |
| **TOTAL** | | 317 | 529 | **117** |

These are legitimate answers (e.g. multiple myeloma is the true indication for 15 anti-BCMA
and related study products in the test set). Popularity adjustment removes the correct answer
for 39 % of the evaluation set; any metric computed after that removal is no longer measuring
performance on those clinically important, common indications.

---

## Analysis 3 — 483 / 201 stratified ESM analysis

**Setup.** The ESM-2 init populated **483** mAb entities with sequence-derived embeddings
(L2 norm ≈ 11.0, min 10.33); the remaining mAbs and all non-mAbs are random
(norm ≈ 0.146, max 0.155). The norm>0.3 threshold separates them cleanly and yields exactly
483 ESM rows (matches `entity_init_metadata.json: num_mab_entities_initialized = 483`); the
other **201** mAb entities (684 total mAbs) were random (112 missing INN link + 89 missing/empty
sequence).

Each of the 297 test query heads (a StudyProduct) was resolved to its mAb via the KG chain
`StudyProduct --isStudyProductOf--> Product --isProductOf--> mAb` (the traversal used in
`reverse_clinical_mab_candidates.py`). **All 297 heads resolved (100 %).** Each query was
labelled **ESM-mAb** or **random-mAb** by that mAb's init status:

- **ESM-mAb stratum: n = 229**
- **random-mAb stratum: n = 68**

For every model (9 architectures × esm2/random × 5 seeds = 90 checkpoints) we computed the
per-query reciprocal rank (standard filtered, no exclusion), averaged within each stratum, and
took the **ESM − random ΔMRR paired by seed** (5 seeds → paired t-test, df = 4). Full table:
`esm_stratified.csv`.

### Pooled across the 9 architectures (mean over seeds)

| stratum | n | MRR esm2 | MRR random | ΔMRR (esm−rand) | paired p |
|---|---|---|---|---|---|
| ESM-mAb   | 229 | 0.3166 | 0.3100 | **+0.0066** | 0.016 |
| random-mAb | 68 | 0.2208 | 0.2089 | **+0.0119** | 0.0009 |
| all        | 297 | 0.2947 | 0.2868 | +0.0078 | 0.002 |

### Per-architecture ΔMRR (esm2 − random, mean over 5 seeds)

| model | Δ ESM-mAb (n=229) | Δ random-mAb (n=68) |
|---|---|---|
| RotE   | +0.0104 (p .009) | +0.0147 (p .004) |
| TransE | +0.0153 (p .002) | +0.0222 (p .027) |
| RefE   | +0.0100 (p .007) | +0.0272 (p .0006) |
| MurE   | +0.0026 (ns) | −0.0004 (ns) |
| AttE   | +0.0015 (ns) | +0.0045 (ns) |
| CP     | +0.0145 (ns, noisy) | +0.0286 (p .03) |
| AttH   | +0.0031 (ns) | +0.0136 (p .02) |
| RefH   | −0.0003 (ns) | +0.0008 (ns) |
| RotH   | +0.0026 (ns) | −0.0043 (ns) |

### Conclusion (Analysis 3)

**The ESM benefit does NOT concentrate on the queries whose mAb actually received an ESM
embedding — so it is not attributable to the per-entity ESM vector; it is a diffuse,
graph-level effect.** Contrary to the "attributable" hypothesis, the ΔMRR is, if anything,
*larger* on the random-mAb stratum (pooled +0.0119, p < 0.001) than on the ESM-mAb stratum
(+0.0066, p = 0.016); in relative terms this is +5.7 % vs +2.1 % over the respective random
baselines. Every Euclidean/translational architecture that shows a real ESM benefit
(RotE, TransE, RefE; also CP) shows it in *both* strata and slightly more in the random-mAb
stratum, while the hyperbolic models (AttH/RefH/RotH) and the additive Euclidean MurE/AttE show
~null ESM effects in both strata.

Interpretation: initialising 483 mAb anchors with ESM structure reshapes the learned geometry
of the shared StudyProduct/disease neighborhood during training; the improvement then
propagates to queries whose own mAb was randomly initialised (these are mAbs that *lacked*
sequence data). The effect is real and significant overall, but it is **systemic rather than
localized** — it cannot be credited to the ESM embedding sitting on the specific queried mAb.

Caveats: the random-mAb stratum has only 68 queries, so per-model estimates there are noisy
(large cross-seed sd for CP/RotH); the pooled paired-by-seed test, however, is robustly
significant. Significance is assessed across 5 seeds (df = 4) treating the seed as the unit of
replication.
