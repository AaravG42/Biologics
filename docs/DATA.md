# Data Provenance and Licensing

This document describes the knowledge-graph data distributed under `data/`,
its provenance, and the terms under which it may be used. **The dataset is
not the original work of this repository's authors.** It is third-party data
derived from the IMGT/mAb-KG / IMGT/mAbOnco-KG resources, redistributed here
only to make the study reproducible, and it carries its own licence and terms
of use (see "Licensing and terms" below).

## Source

The data originates from the IMGT/mAb-KG and IMGT/mAbOnco-KG work of
Sanou et al.:

- **Sanou et al. (2026)** — "Therapeutic monoclonal antibodies repurposing in
  oncology via IMGT/mAb-KG embeddings", *BMC Medical Informatics and Decision
  Making* 26:89. doi:[10.1186/s12911-026-03394-4](https://doi.org/10.1186/s12911-026-03394-4)
- **Sanou et al. (2024)** — "IMGT/mAb-KG: The knowledge graph for therapeutic
  monoclonal antibodies", *Frontiers in Immunology* 15:1393839.
  doi:[10.3389/fimmu.2024.1393839](https://doi.org/10.3389/fimmu.2024.1393839)

The source dataset repository is
`https://src.koda.cnrs.fr/imgt-igh/oncomabkgembeddings`. The underlying
biological data is curated by **IMGT**, the international ImMunoGeneTics
information system (<https://www.imgt.org>).

## Dataset directories

All three directories share the **same entity vocabulary of 9,611 entities**.
They differ only in how the triples are split and which relations are retained.

### `data/IMGT`
A **byte-identical copy** of Sanou et al.'s processed IMGT/mAbOnco-KG
train/valid/test splits. This is the unmodified baseline split, included so
the upstream data and our re-splits can be compared directly.

| split | triples |
|-------|--------:|
| train | 24,135  |
| valid |  2,856  |
| test  |  2,804  |

### `data/IMGT_no_leakage`
A **reciprocal-leakage-removed re-split** of the same triples. The base
IMGT/mAbOnco-KG split allows a fact and its reciprocal (inverse) to appear in
different splits, which lets a model trivially recover a held-out test fact
from its inverse seen during training. This re-split reorganizes the triples
so that reciprocal pairs are kept together, removing that source of leakage.
The overall split sizes are unchanged from `data/IMGT`:

| split | triples |
|-------|--------:|
| train | 24,135  |
| valid |  2,856  |
| test  |  2,804  |

### `data/IMGT_no_leakage_directional_clinical`
Built on top of `IMGT_no_leakage`, but with the **inverse clinical relation
`imgt:isClinicalIndicationOf` dropped** so that the clinical-indication
relation is modeled in a single direction only (`imgt:hasClinicalIndication`).
This prevents the model from answering a `hasClinicalIndication` query by
memorizing the corresponding inverse triple. Dropping the inverse relation
reduces the triple counts:

| split | triples |
|-------|--------:|
| train | 22,921  |
| valid |  2,555  |
| test  |  2,507  |

In this directional split, the **test set contains 297
`imgt:hasClinicalIndication` triples and 0 inverse
(`imgt:isClinicalIndicationOf`) triples**, which is the set of clinical
repurposing queries evaluated in the study.

## File format

Each split directory contains tab/whitespace-separated triple files
(`train`, `valid`, `test`) plus the pre-computed `*.pickle` and
`to_skip.pickle` index files consumed by the data loaders.

## Licensing and terms

- The **source dataset repository** (`oncomabkgembeddings`) is licensed under
  the **European Union Public Licence v1.2 (EUPL-1.2)**.
- The **underlying data** is from **IMGT** and is subject to the IMGT terms of
  use (<https://www.imgt.org>).

These terms are **independent of, and take precedence over**, the Apache-2.0
licence that applies to this repository's *code*. The data is redistributed
here under those original terms.

We make **no claim of ownership** over the IMGT/mAbOnco-KG data. If you use the
data, you must comply with the EUPL-1.2 licence of the source repository and
the IMGT terms of use, and you must cite Sanou et al. (2024, 2026) and IMGT.
See the repository `NOTICE` file for the full attribution.
