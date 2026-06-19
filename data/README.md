# Data

This directory holds the knowledge-graph splits used in the study. The data is
**third-party data** derived from the IMGT/mAb-KG / IMGT/mAbOnco-KG resources
of Sanou et al. and IMGT — it is **not** the original work of this repository's
authors and carries its own licence (EUPL-1.2 / IMGT terms).

For full provenance, triple counts, and licensing, see
[`../docs/DATA.md`](../docs/DATA.md).

## Directories

- **`IMGT/`** — byte-identical copy of Sanou et al.'s processed
  train/valid/test split (24,135 / 2,856 / 2,804).
- **`IMGT_no_leakage/`** — reciprocal-leakage-removed re-split of the same
  triples (24,135 / 2,856 / 2,804).
- **`IMGT_no_leakage_directional_clinical/`** — as above, with the inverse
  `imgt:isClinicalIndicationOf` relation dropped so clinical indication is
  modeled in one direction only (22,921 / 2,555 / 2,507). The test set holds
  297 `imgt:hasClinicalIndication` queries and 0 inverse triples.

All three share the same 9,611-entity vocabulary.

See [`../docs/DATA.md`](../docs/DATA.md) for citations and the full terms of
use before redistributing or reusing this data.
