# Provenance & Verification Table

This document reconciles **every dataset count and model artifact** reported in
the manuscript *"Repurposing Therapeutic Monoclonal Antibodies in Oncology Using
a Pharmacologically Validated Knowledge Graph–Based Framework"* (British Journal
of Clinical Pharmacology, Submission ID `5f071cd3-9d74-4074-bdc9-95754b548f4d`)
against the code, data, and result files in this repository. It is intended to
let a reviewer independently confirm that the reported experiments correspond to
artifacts actually present here.

Numbers below are **regenerated from the committed artifacts** — not transcribed
from the manuscript. The reproduction script
[`results/tables/table_s3_reproduced.csv`](../results/tables/table_s3_reproduced.csv)
is produced directly from the per-trial `metrics.json` files under
`results/checkpoints/`.

---

## 1. Knowledge graph & dataset counts (manuscript Table 1 / S1)

| Quantity | Value | Where to verify |
|---|---|---|
| Entities | 9,611 | `esm_init/entity_init_metadata.json` (`num_entities`); `wc -l` on entity files |
| Directed triples | 29,795 | `data/IMGT` (train+valid+test) |
| Distinct relations | 53 (base) / 52 (directional, inverse dropped) | `esm_init/entity_init_metadata.json` |
| `data/IMGT` split (train/valid/test) | 24,135 / 2,856 / 2,804 | `wc -l data/IMGT/{train,valid,test}` |
| `data/IMGT_no_leakage` split | 24,135 / 2,856 / 2,804 | `wc -l data/IMGT_no_leakage/{train,valid,test}` |
| `data/IMGT_no_leakage_directional_clinical` split | 22,921 / 2,555 / 2,507 | `wc -l data/IMGT_no_leakage_directional_clinical/{train,valid,test}` |
| Clinical-indication test triples (forward) | 297 | `grep -c imgt:hasClinicalIndication data/IMGT_no_leakage_directional_clinical/test` |
| Inverse `isClinicalIndicationOf` in directional | 0 | `grep -c imgt:isClinicalIndicationOf data/IMGT_no_leakage_directional_clinical/*` |

The base `data/IMGT` split is a **byte-identical copy** of Sanou et al.'s
processed IMGT/mAbOnco-KG split (see [`DATA.md`](DATA.md)); it is **not** re-split
here. `IMGT_no_leakage` and `IMGT_no_leakage_directional_clinical` are produced
from it by `scripts/create_imgt_no_leakage.py` and
`scripts/create_imgt_directional_clinical.py`.

## 2. Reciprocal data leakage (manuscript Table 2 / S2)

The leakage audit (80.3% of validation, 82.7% of forward test, 86.4% of inverse
test clinical triples having their reciprocal in train; 7,390 total cross-split
reciprocal links) can be re-derived from the original `data/IMGT` split with
`scripts/create_imgt_no_leakage.py`, which reports the reciprocal counts it
removes.

## 3. KGE benchmark — Table 3 / S3 reconciliation

Each manuscript Table S3 row maps to one HPO trial, selected by **lowest
validation forward-clinical tail Mean Rank** (the manuscript's stated selection
rule). The per-trial config and metrics are committed under
`results/checkpoints/<MODEL>_<trial>/`; the full HPO record for **all 1,661
trials across the benchmarked models** is in `results/hpo_all_trials.csv`, and
the auto-generated `summary.json` is in `results/`.

| Model | Backing trial | Checkpoint dir | Reproduces Table S3? |
|---|---|---|---|
| RotE | trial_101 | `results/checkpoints/RotE_trial_101/` (incl. `model.pt`) | ✔ via `metrics.before_hardneg.json` — see ⚠ note below |
| TransE | trial_079 | `results/checkpoints/TransE_trial_079/` | ✔ exact |
| RefE | trial_065 | `results/checkpoints/RefE_trial_065/` | ✔ exact |
| MurE | trial_163 | `results/checkpoints/MurE_trial_163/` | ✔ exact |
| AttE | trial_154 | `results/checkpoints/AttE_trial_154/` | ✔ exact |
| CP | trial_099 | `results/checkpoints/CP_trial_099/` | ✔ exact |
| AttH | trial_000 | `results/checkpoints/AttH_trial_000/` | ✔ exact |
| RotH | trial_031 | `results/checkpoints/RotH_trial_031/` | ✔ exact |
| RefH | trial_059 | `results/checkpoints/RefH_trial_059/` | ⚠ MRR/Hits exact; **MR differs** — see below |

To keep the committed tree small, only the RotE checkpoint ships its `model.pt`
(it is the model used for all downstream interpretability and repurposing
analysis). Every model ships its full `config.json` + `metrics.json`, and any
checkpoint can be regenerated with `scripts/hpo_imgt_clinical_directional.py`
(see [`REPRODUCE.md`](REPRODUCE.md)); seeds are deterministic
(`seed = 42 + 1000*(model_index+1) + trial_index`).

### ⚠ Two artifact/manuscript discrepancies (disclosed for full transparency)

1. **RotE checkpoint state.** The manuscript's RotE numbers (MR 24.96, MRR
   0.316, Hits@1 19.5%, Hits@3 34.68%, Hits@10 59.26%) correspond to trial_101
   **before** a subsequent hard-negative fine-tuning step. That fine-tuning
   overwrote `model.pt`; the committed `model.pt` is the **post-fine-tuning**
   state (MR 24.56, MRR 0.311). Both metric snapshots are committed
   (`metrics.before_hardneg.json` = manuscript numbers; `metrics.json` =
   shipped checkpoint). The five interpretability case studies (§5) were
   computed on the **pre-fine-tuning** predictions
   (`interpretability/artifacts/interpret_topk_50.json`). A clean, fixed-seed,
   multi-seed re-run is the recommended fix and is tracked for the next revision.

2. **RefH Mean Rank.** RefH's MRR/Hits@1/Hits@3/Hits@10 (0.284 / 17.8% / 30.6% /
   50.2%) match trial_059 exactly, but that trial's **test Mean Rank is 429.63,
   not the 38.22 printed in Table S3** — i.e. the Table S3 RefH MR appears to be
   a transcription error in the manuscript. The repository value is the correct
   one.

## 4. ESM-2 entity initialization (manuscript §2.4)

`esm_init/entity_init_metadata.json` records exactly: 9,611 entities; init
dimension 640 (additive fusion `add(first_half, second_half)` of the 1,280-d
ESM-2 source); **483 of 684 mAb entities initialized**, 112 lacking an INN link,
89 lacking a usable sequence; all non-mAb entities random
`uniform(-0.01, 0.01)`. The manuscript's "ESM-2 reduced Mean Rank by ~10" result
is reported only on the **original (pre-leakage-removal)** dataset; a clean-data,
repeated-run characterization is **not** included in the submission and is
flagged for the next revision.

## 5. Pharmacological validation — case studies (Tables S6–S11)

| Manuscript item | Artifact |
|---|---|
| Top-50 predictions per query (RotE) | `interpretability/artifacts/interpret_topk_50.json` |
| Five case studies (S6–S10): PD-1/MSLN/HER2/EGFR/BCMA, PLAUSIBLE/OFF-TARGET/GENERIC tags | `interpretability/artifacts/interp_case_top20.json`, `interp_plausibility.json` |
| Open Targets target–disease associations | `interpretability/artifacts/gene_disease_assoc.json`, `cache_disease2targets.json` |
| Enrichment table (1.8× over prior) | `interpretability/artifacts/indication_prior.json` + the case files |
| Reactome over-representation, anti-PD-1 10 genes (Table S11) | `interpretability/artifacts/pd1_case_reactome.json` |
| Reverse (disease→mAb) repurposing inference | `interpretability/artifacts/reverse_imgt_Chronic_lymphocytic_leukemia_CLL.json` |
| Mechanism-of-action UMAP | `interpretability/artifacts/mechanism_umap_final.{png,json}` |

The top Reactome pathway ("Constitutive Signalling by Aberrant PI3K in Cancer")
has FDR `1.2197864e-07` in `pd1_case_reactome.json` — i.e. **1.2 × 10⁻⁷**,
matching the main text. Any "1.2 × 10⁻⁶" rendering in Figure 3 is an error and
should read 10⁻⁷.

The biological verification was performed with the external **celltype-agent**
tool calling the **public** Open Targets GraphQL API, Reactome AnalysisService,
and MyGene.info; the committed artifacts are an **April-2026 snapshot**. The
plausibility tagging in the submission was applied to **5 hand-selected cases**
and was **not** blinded or run across all 297 queries — a systematic, blinded
re-scoring with a pre-registered threshold is planned for the next revision (see
[`../interpretability/README.md`](../interpretability/README.md)).

## 6. Known issues to correct in any resubmission

These are documented here so the repository does not silently contradict the
submitted manuscript:

- RotE selected as "best" by standard MRR, but on the popularity-adjusted metric
  the manuscript itself prefers, RefH and MurE rank higher; all case studies
  nonetheless use RotE.
- The split is **inductive** (clinical head entities disjoint across splits),
  which does not match the **transductive** controlled-split protocol of Brière
  et al. that the manuscript cites alignment with.
- Reference [13] (Brière et al.) DOI resolves to an unrelated paper; the citation
  must be corrected.
- The data source is **IMGT/mAb-KG** (Sanou et al.), not "IMGT/mAb-DB"; the
  benchmark, baselines, and interpretability approach build on Sanou et al. and
  are credited as such here and in [`DATA.md`](DATA.md).
- "Evaluate foundation models" in the contribution list refers to ESM-2
  initialization only; no foundation model is developed.

See the top-level [`README.md`](../README.md) for how this repository relates to
the prior work of Sanou et al.
