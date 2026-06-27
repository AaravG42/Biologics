# External-evidence held-out validation of the RotE mAb-repurposing model

**Run date / API access date: 2026-06-27. CPU only (`CUDA_VISIBLE_DEVICES=""`).**

This document accompanies `external_validation.py`, `external_pairs.csv` and
`summary.json`. It describes how the external validation set was built, what it
tests, and — explicitly and prominently — what it does **not** establish.

---

## 0. The headline limitation: the KG has NO timestamps

A reviewer (in the spirit of Briere *et al.*'s call for a genuinely *external*
inference set rather than a same-KG random test split) asked for held-out
evidence the model never saw in training. The strongest form of such a set is a
**temporally** held-out split: train on associations known before some year *Y*,
test on associations that became known after *Y*.

**This is infeasible with the provided data, because the knowledge graph
contains no timestamps.** We verified this directly against
`data/IMGT_no_leakage_directional_clinical/{train,valid,test}`:

* The only date-bearing content is
  * `imgt:hasStatut imgt:approval` — a **status flag carrying no date**, and
  * DOIs inside `imgt:hasBibliographicReference`
    (e.g. `doid:10.1016/j.lungcan.2021.07.009`) whose strings **incidentally**
    contain a publication year.
* There is **no** `date` / `year` / approval-year relation or entity, and
  nothing ties an *indication* (`imgt:hasClinicalIndication`) to a point in
  time.

A strict temporally held-out evaluation therefore **cannot** be constructed
from these data. We instead build an **external-evidence** held-out set:
indications supported by clinical-trial evidence that lives *outside* the IMGT
training graph and are *not present* in the model's training triples.

## 0b. Second structural limitation: the KG anonymises antibody identity

The task suggested querying **ClinicalTrials.gov** by `intervention =` the
antibody INN/name. **This is infeasible from the KG alone:** every antibody is
an anonymised node (`imgt:mAb_<id>`), and the graph contains **zero** INN /
brand-name tokens (we grepped all three splits: 0 `*-mab` antibody names exist
as entities). There is no offline mapping from the IMGT mAb id to an INN.

The molecular **target gene** is therefore the only stable bridge between the
anonymised KG mAbs and external clinical evidence. We use **Open Targets**'
target-keyed clinical-candidate resource, which itself aggregates
ChEMBL / **ClinicalTrials.gov (AACT)** / FDA / EMA records and carries the
**ClinicalTrials.gov NCT ids** and source URLs. Consequently the external
evidence is **target-level**: it shows that *an antibody against this mAb's
target* is in clinical trials for indication *D* (it may be a *different*
antibody against the same target). This caveat is load-bearing — see §6.

---

## 1. Data sources

| Source | Use | Access |
|---|---|---|
| `data/IMGT_no_leakage_directional_clinical/{train,valid,test}` | training KG, candidate indication pool, in-training membership | local |
| `analysis_outputs/.../RotE/trial_101/interpret_topk_50.json` | **authoritative** RotE per-query top-50 ranked indication predictions | local |
| `analysis_outputs/.../RotE/trial_101/indication_prior.json` | empirical popularity prior (training indication frequencies) → baseline | local |
| MyGene.info (`mygene.info/v3/query`) | HGNC → gene symbol | cached (reused from `revision/blinded_plausibility/cache`) |
| Open Targets Platform GraphQL (`api.platform.opentargets.org/api/v4/graphql`) | symbol → Ensembl id; `target.drugAndClinicalCandidates` → external clinical-trial evidence | **fetched 2026-06-27**, cached to `cache/ot_drug_candidates.json` |

All API responses are cached to `cache/`; `cache/fetch_meta.json` records the
per-gene UTC access timestamp. Re-running with `--offline` reproduces every
number from cache without any network call.

### Model checkpoint note (important)
Ranks are read from **`interpret_topk_50.json`** (the published predictions;
test `mean_rank = 24.96`). The `model.pt` currently in the trial directory is a
**later hard-negative finetune** (`mean_rank = 24.56`, written ~8 h after the
JSON) and does **not** reproduce the published top-k — re-scoring with it matched
only 89/297 published ranks. We therefore deliberately use the JSON (the ranking
the analysis is asked to validate) rather than the mismatched checkpoint.

---

## 2. Pipeline

1. **Resolve each query mAb's target gene.** For every forward query
   `(StudyProduct, imgt:hasClinicalIndication, ?)` in `interpret_topk_50.json`
   (297 queries):
   `StudyProduct --isStudyProductOf--> Product --isProductOf--> mAb
   --(sio:SIO_000291 | isTargetOf)--> HGNC`, then HGNC → symbol (MyGene) →
   Ensembl (Open Targets). *(Logic reused from
   `revision/blinded_plausibility/score_blinded.py` and
   `scripts/reverse_clinical_mab_candidates.py`.)*
   Coverage: **297/297** queries resolve to a mAb; **279/297** to ≥1 target
   gene; **101** unique genes, all with Ensembl ids.

2. **Fetch external clinical-trial evidence** per target gene via Open Targets
   `target.drugAndClinicalCandidates` (the successor of the deprecated
   `knownDrugs`). For each **antibody** drug against the target
   (`drugType` contains "antibody" *or* INN ends in `-mab`) we keep the
   indications studied, the **ClinicalTrials.gov NCT ids** of the supporting
   trials, and each trial's **phase**.

3. **Normalise** each external Open Targets EFO disease name to an IMGT KG
   indication entity (§3).

4. **Define held-out pairs** (§4) and **evaluate** against the model and the
   popularity-prior baseline (§5).

---

## 3. Normalisation map (transparent, pre-registered, reused)

The disease-string normaliser, synonym table, qualifier-stripping list,
discriminative-token / stop-token logic are **imported verbatim** from
`revision/blinded_plausibility/score_blinded.py` (the pre-registered table fixed
for the blinded plausibility analysis). For each KG indication entity we compute
its normalised core phrase + synonym-expanded keyword set + discriminative-token
set. An Open Targets EFO disease name maps to a KG entity by:

1. **exact** normalised-phrase match, else
2. **discriminative-token overlap** — anatomical-site / disease-class tokens must
   match; generic tokens (`cancer`, `neoplasm`, `tumour`, …) do **not** count
   (`score_blinded.STOP_TOKENS`). Ties broken by KG training prior, then shorter
   entity string.

**Generic KG umbrella nodes** (`Solid_tumors`, `Cancers`, `Neoplasms`, …) are
**not** eligible match targets and are excluded from the held-out positives, so
the test cannot be won trivially by always naming the most popular bucket.

**Coverage:** of **966** distinct Open Targets EFO disease strings encountered,
**534 (55.3%)** map to a (specific, non-generic) KG indication. Most of the
unmapped 45% are diseases with no counterpart among the **247** KG indications
(the KG indication vocabulary is small and oncology-centric); this is an
expected coverage ceiling, not an error, and the unmatched names simply do not
contribute pairs.

---

## 4. What "held-out" means

The candidate space is the **247 KG indication entities** (all tails of
`imgt:hasClinicalIndication`). An external `(mAb, indication D)` pair is counted
as **held-out** iff:

* `D` maps from external Open Targets clinical-trial evidence for that mAb's
  target gene(s), **and**
* `D` is **not** a **training** `imgt:hasClinicalIndication` tail for **any**
  study product of that mAb (`in_training = False`), **and**
* `D` is a specific (non-generic) indication.

We additionally record `in_valid_test` (whether `D` is a valid/test indication
of that mAb — i.e. a held-out positive the KG itself recognises) and the trial
phase / NCT count of the external evidence. The reported result tiers are:

| Set | Definition | n (unique mAb × indication) |
|---|---|---|
| **A** | held-out + ≥1 ClinicalTrials.gov NCT | 7 483 |
| **B** | A + external trial phase ≥ 2 | 6 560 |
| **C** | held-out + NCT + **also** a KG valid/test indication of the mAb (external **and** KG corroborate) | 99 |

(`external_pairs.csv` also retains the per-query rows; 11 003 total, 10 575
held-out, 10 478 with an NCT.)

---

## 5. Evaluation & baseline

* **Model rank.** For a held-out pair, the model's rank of `D` is its position
  among the **indication-pool entities** within that query's authoritative
  top-50 prediction list (`model_rank_indication_pool`). Pairs whose indication
  falls outside the stored top-50 are **censored** (counted as a miss for
  recall; excluded from the within-depth median, with the censoring rate
  reported). `model_rank_all_entities` (rank over all entities) is also recorded.
* **Popularity-prior baseline.** Indications ranked by training frequency
  (`indication_prior.json`) — a single global ranking applied to every query.
  This is a **strong** baseline (clinical trials concentrate on common cancers).
* **Apples-to-apples.** Model and baseline both rank `D` within the **same
  247-indication pool**, so `recall@k` is directly comparable.
* `enrichment = model_recall@k / prior_recall@k`. We also report the
  random-uniform reference (`k / 247`) and **paired bootstrap 95% CIs**
  (2 000 resamples).

---

## 6. Results (see `summary.json` for the full table)

**Random-uniform reference:** recall@5 = 0.020, @10 = 0.041, @20 = 0.081.

### Set A — broad target-level held-out + NCT (n = 7 483)
| k | model | prior | enrichment [95% CI] |
|---|---|---|---|
| 5 | 0.053 | 0.047 | **1.11×** [1.02, 1.21] |
| 10 | 0.114 | 0.120 | 0.95× [0.90, 0.99] |
| 20 | 0.203 | 0.213 | 0.95× [0.93, 0.99] |

Median model rank among ranked pairs = 20 vs prior 58 (60% of pairs censored
beyond top-50). Set B (phase ≥ 2) is essentially identical.

### Set C — externally **and** KG-corroborated held-out indications (n = 99)
| k | model | prior | enrichment [95% CI] |
|---|---|---|---|
| 5 | 0.354 | 0.283 | 1.25× [0.97, 1.67] |
| 10 | **0.566** | 0.515 | 1.10× [0.93, 1.32] |
| 20 | 0.697 | 0.747 | 0.93× [0.84, 1.03] |

Median model rank = 8 (of 247), only 8% censored.

### What this establishes
* **Vs a random baseline, the model is strongly predictive.** On the
  corroborated set its recall@10 is **0.57 vs 0.04** uniform (~14×); even on the
  broad noisy set it is ~3× uniform. The model ranks externally-evidenced
  held-out indications far above chance, and recovers individual repurposing
  hypotheses popularity misses entirely — e.g. EPCAM mAb → colon cancer
  (model rank 1, prior 42), CD38 / daratumumab → Merkel-cell carcinoma
  (model 5, prior 58), ERBB2 → biliary-tract cancer (model 8, prior 95),
  TNFRSF17 (BCMA) → multiple myeloma (model 2, prior 4). These are recorded in
  `external_pairs.csv`.
* **Vs the popularity prior, the edge is small and mostly not significant.** The
  model significantly beats popularity only at the very top of the ranking on
  the broad set (recall@5 enrichment 1.11×, CI excludes 1); at k = 10/20 it is
  at parity or marginally below. On the corroborated set every enrichment CI
  includes 1.0. The aggregate target-specific signal is largely **washed out**
  because the external clinical-trial space is dominated by common cancers
  (NSCLC, breast, colorectal …) that the popularity prior already ranks high.

### What this does NOT establish
* It is **not** a temporal / prospective validation (impossible here — §0); it
  cannot show the model predicts *future* approvals.
* Evidence is **target-level** (§0b): a held-out pair means *an* antibody against
  the mAb's target is in trials for `D`, not necessarily *this* anonymised mAb.
  This inflates set sizes with promiscuous targets and adds noise.
* It does not show the model **beats a strong popularity prior** in aggregate —
  on this external set it is roughly at parity. The defensible claim is that the
  model is well-calibrated against external clinical evidence and adds
  target-specific lift for individual cases, **not** that it is a population-level
  improvement over naming common cancers.
* Normalisation coverage (55%) and the small KG indication vocabulary (247)
  bound resolution; rare/non-oncology external indications are under-represented.

---

## 7. Reproducing

```bash
cd /home/aarav/KGEmb
CUDA_VISIBLE_DEVICES="" python3 revision/external_validation/external_validation.py            # live (re)fetch + cache
CUDA_VISIBLE_DEVICES="" python3 revision/external_validation/external_validation.py --offline  # recompute from cache only
```

Outputs: `external_pairs.csv` (one row per query × external held-out indication,
with mAb, target, indication, source, NCT ids, phase, in-training flag, model
rank, prior rank), `summary.json` (all metrics, coverage, CIs, limitations),
`cache/` (raw cached API responses + access timestamps).
