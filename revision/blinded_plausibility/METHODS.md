# Blinded biological-plausibility scoring — pre-registered methods

This directory implements a **systematic, blinded, reproducible** biological-
plausibility scorer for the RotE knowledge-graph-embedding model's forward
clinical-indication predictions. It exists to answer a reviewer criticism that
the original paper's plausibility verdicts were assigned to only **5 hand-
selected cases, unblinded, with manual overrides**.

Here the same plausibility rule is **pre-registered** (fixed as code constants
before any of the 297-query results were inspected) and applied **identically
and blindly** to **all 297** forward `(StudyProduct, imgt:hasClinicalIndication,
?)` queries. The tagging logic never reads the ground-truth tail or the model's
rank of that tail; it sees only (a) the model's top-20 predicted indications and
(b) the query antibody's target gene(s).

## Inputs (read-only)

| File | Role |
|------|------|
| `analysis_outputs/.../RotE/trial_101/interpret_topk_50.json` | per-query rank + top-50 predicted indications (297 forward queries) |
| `analysis_outputs/.../RotE/trial_101/indication_prior.json` | training-set indication frequencies (popularity prior) |
| `data/IMGT_no_leakage_directional_clinical/{train,valid,test}` | KG triples for target resolution |
| `analysis_outputs/.../RotE/trial_101/cache_hgnc2symbol.json` | HGNC→symbol (re-used as a seed cache) |
| `analysis_outputs/.../RotE/trial_101/gene_disease_assoc.json` | original 5-case Open Targets pulls (used **only** to seed/validate, never as input to the 297 scoring) |
| `analysis_outputs/.../RotE/trial_101/interp_plausibility.json` | original 5-case tags (used **only** for the validation note) |

## Target resolution (KG traversal)

For each query `StudyProduct` we traverse:

```
StudyProduct --imgt:isStudyProductOf--> Product
             --imgt:isProductOf-------> mAb
             --sio:SIO_000291---------> HGNC:id      (also accept reverse  HGNC --imgt:isTargetOf--> mAb)
```

`HGNC:id` is resolved to an approved gene symbol via **MyGene.info**
(`https://mygene.info/v3/query`, scopes=HGNC, fields=symbol). All 297 queries'
HGNC ids were already present in the project's HGNC cache, so no live MyGene
calls were required in practice; the code path remains and caches any new
lookups to `cache/hgnc2symbol.json`.

Antibodies may have **more than one** target gene (33/297 queries are
multi-target); the Open Targets associated-disease lists of all target genes are
**unioned** before scoring. Queries whose antibody has **no HGNC target**
(e.g. anti-ganglioside / anti-glycan antibodies — 18/297) are reported but left
unscorable.

## Open Targets associations

Gene symbol → Ensembl gene id via the Open Targets `search` GraphQL field
(exact `approvedSymbol` match preferred). Ensembl id → associated diseases via
the `associatedDiseases` field, retaining the **top 50** diseases per gene with
their overall association `score`. Endpoint:
`https://api.platform.opentargets.org/api/v4/graphql`.

All API responses are cached on disk (`cache/symbol2ensembl.json`,
`cache/gene_ot_assoc.json`) so the run is fully **repeatable and resumable**;
the script sleeps `API_SLEEP` seconds between live calls to be rate-limit
friendly. The Open Targets Platform API snapshot used is **v26.03**
(`meta.apiVersion` = 26.03.1, data version 26.03), recorded in `summary.json`
under `api_snapshot`.

## PRE-REGISTERED SCORING RULE (fixed before any result was seen)

Constants in `score_blinded.py`:

```
PLAUSIBILITY_THRESHOLD = 0.01          # primary OT association-score threshold
SENSITIVITY_THRESHOLDS = [0.01, 0.05, 0.1]
TOP_K                  = 20            # rank window scored per query
N_PRIOR_DRAWS          = 1000          # Monte-Carlo replicates for the baseline
PRIOR_DRAW_SIZE        = 20
RANDOM_SEED            = 20240617      # deterministic baseline
OT_DISEASES_PER_GENE   = 50
```

For each query, over its **top-20** predicted indications, each prediction is
classified:

- **GENERIC** — the IMGT string is a non-specific umbrella/bucket node
  (`Solid tumors`, `Cancers`, `Neoplasms`, `Oncology`, `Tumors`,
  `Malignancies`, `Hematologic malignancies`, …). Generic predictions are
  **excluded from the denominator** (a plausibility verdict on an
  uninterpretable bucket would be meaningless). The test is on the **full**
  indication phrase, so qualified variants such as
  `Solid tumors advanced or metastatic` remain **specific** (and tag
  OFF-TARGET), matching the original hand-tagging which counted them in the
  denominator.
- **PLAUSIBLE** — the normalised disease shares a **discriminative**
  anatomical-site / disease-lineage token (or multiword phrase) with an Open
  Targets disease that is associated with the target gene at
  `score >= PLAUSIBILITY_THRESHOLD`.
- **OFF-TARGET** — specific indication with no qualifying Open Targets
  association.

Per query: `model_fraction = #plausible / (#plausible + #off-target)` over the
specific (non-generic) top-20.

### Normalisation map (transparent, fixed in advance)

`normalise_disease()` turns an IMGT entity string into a set of comparable
phrases/tokens:

1. lowercase; strip the `imgt:` prefix; replace `_` with spaces; remove
   possessive `'s`.
2. capture parenthetical abbreviations (e.g. `(NSCLC)`), then remove parens.
3. **expand abbreviations** via the `SYNONYM_TABLE` (e.g. `NSCLC →
   non-small cell lung carcinoma`, `HNSCC → head and neck squamous cell
   carcinoma`, `CRC → colorectal carcinoma`, `NHL → non-hodgkin lymphoma`,
   `CLL`, `AML`, `MM`, `TNBC`, `HCC`, `DLBCL`, …; full table in the source).
4. **strip qualifier phrases** (`metastatic`, `recurrent`, `advanced`,
   `relapsed or refractory`, `overexpressing ERBB2`, line-of-therapy/stage
   context, …) to obtain a clean core term.
5. reorder organisational `Cancers_<site>` → `<site> cancer`.

### Matching (token overlap, generic-aware)

A prediction matches an OT disease iff they share at least one **discriminative**
token or multiword phrase **and** the OT score ≥ threshold. Pure umbrella tokens
(`cancer`, `neoplasm`, `tumor`, `malignant`, …) and bare histology suffixes
(`carcinoma`, `adenocarcinoma`, `squamous`, `large`) are in a **stop-token** set
and do **not** by themselves create a match — this prevents every specific
tumour from spuriously matching Open Targets' "cancer"/"neoplasm" umbrella
entries. Anatomical-site tokens (`gastric`, `breast`, `ovarian`, `lung`,
`pancreatic`, …) and lineage tokens (`lymphoma`, `leukemia`, `melanoma`,
`mesothelioma`, `sarcoma`, …) are discriminative, as are multiword phrases such
as `non-small cell lung`, `head and neck`, `diffuse large b-cell`,
`triple-negative breast`.

## Popularity baseline (Monte-Carlo)

For each query we draw `PRIOR_DRAW_SIZE = 20` indications **with replacement**,
weighted by their empirical training-set frequency
(`indication_prior.json["counts"]`), score them with the **same rule** against
the **same** target-gene OT list, and average over `N_PRIOR_DRAWS = 1000` draws
to obtain `prior_fraction`. `enrichment = model_fraction / prior_fraction`. The
seed (`RANDOM_SEED`) makes the baseline deterministic. To avoid re-normalising
the prior thousands of times per query, every prior indication is pre-classified
once per distinct gene-set.

## Outputs

- `results_297.csv` — per query: mAb, target gene(s), `#specific`, `#plausible`,
  `model_fraction`, `prior_fraction`, `enrichment`, `#unmatched`, `status`.
- `summary.json` — aggregate over all scored queries: mean/median model
  plausible-fraction, mean/median baseline fraction, mean/median enrichment with
  bootstrap 95% CI, fraction of queries with enrichment > 1, normalisation
  coverage (fraction of specific predictions that matched an OT disease), and
  threshold sensitivity (0.01 / 0.05 / 0.1).
- `validation.json` (when run with `--validate-only`) and the
  `validation_5case` block in `summary.json`.

## Reproducing the run

```bash
python3 score_blinded.py            # full run; live API for any uncached gene
python3 score_blinded.py --offline  # use on-disk caches only (no network)
python3 score_blinded.py --validate-only   # 5-case validation only
```

The run is resumable: every API response is written to `cache/` immediately, so
re-running after an interruption only fetches what is missing.

## Validation against the original 5-case tags

Running the blinded auto-tagger on the original 5 cases reproduces the original
plausible/specific counts closely (target counts from the original manuscript in
parentheses):

| Gene | Blinded (this tool) | Original |
|------|--------------------:|---------:|
| PDCD1 (PD-1)   | 10/17 | 9/17 |
| MSLN           | 12/18 | 10/18 |
| ERBB2 (HER2)   | 9/17  | 9/17 |
| EGFR           | 10/18 | 9/18 |
| TNFRSF17 (BCMA)| 16/18 | 14/18 |

The **specific (denominator) counts match exactly** in all five cases. The
blinded plausible counts are within +0 to +2 of the original hand-tags. The
small upward differences are, on inspection, cases where the **original
auto-tagger under-counted because of string mismatches that a manual read had
already flagged as plausible** — most notably PD-1 position 4
`HNSCC (recurrent/metastatic)`, which the original manuscript explicitly notes
"is a canonical PD-1 indication; keyword mismatch in our auto-tagger" and counts
as plausible on manual read. The remaining differences are lineage-level matches
(e.g. an `ALL`/`CLL` prediction matching the gene's `acute myeloid leukemia`
association via the shared `leukemia` lineage), which are biologically
defensible and applied **uniformly** to all 297 queries rather than overridden
case by case. The point of this exercise is a **consistent, pre-registered,
blinded** rule, not bit-for-bit reproduction of a noisy manual tagging.

## Limitations

- **Unmatched diseases.** Some specific IMGT indications have no Open Targets
  association for the target gene (genuine off-targets) and some fail string
  normalisation; both are counted as OFF-TARGET. The per-query `#unmatched`
  column and the aggregate `normalisation_coverage` quantify this.
- **OT top-50 truncation.** Only the top-50 OT diseases per gene are retained.
  At the pre-registered 0.01 threshold this is effectively "appears among the
  gene's strongest associations"; for some genes the 50th association already
  sits above 0.01, so a few genuine associations beyond rank 50 are not seen.
- **API snapshot.** Results reflect a single Open Targets Platform snapshot
  (version recorded in `summary.json`); scores drift between releases.
- **Multi-target antibodies.** Target-gene OT lists are unioned, which is
  permissive (a prediction plausible for *either* target is counted plausible).
- **No-target antibodies.** Anti-glycan / anti-ganglioside antibodies have no
  HGNC target and are left unscorable.
