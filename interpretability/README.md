# Interpretability and Biological Validation

This directory holds the biological-validation outputs for the best model used
in the study — a **RotE** model (run `trial_101`) — applied to the directional
clinical split. The artifacts under `artifacts/` document the model's
top predictions and the biological plausibility checks performed on them.

## Important caveats (please read)

This biological verification was performed with an **external `celltype-agent`
tool** that queried public web services:

- the **Open Targets** GraphQL API,
- the **Reactome** AnalysisService, and
- **MyGene.info**.

The results are an **April-2026 snapshot**. Because these are **live external
APIs**, re-running the same queries today may return different results. The
scoring captured here was **not** a blinded, systematic re-scoring of every
query: a **systematic, blinded re-scoring across all 297 clinical queries is
planned** but has not yet been completed. The tags and case studies below
should be read with these limitations in mind and **not overstated**.

## Artifacts (`artifacts/`)

- **`interpret_topk_50.json`** — per-query top-50 predictions from the RotE
  `trial_101` model (the raw ranked candidate lists).

- **`interp_plausibility.json`** and **`interp_case_top20.json`** — the five
  case studies (S6–S10). Each candidate is tagged against Open Targets
  evidence as **PLAUSIBLE**, **OFF-TARGET**, or **GENERIC**.

- **`gene_disease_assoc.json`** and **`cache_disease2targets.json`** — Open
  Targets target–disease associations retrieved via the
  `target.disease_association` query (and the cached disease→targets lookups
  backing it).

- **`pd1_case_reactome.json`** — Reactome over-representation analysis for the
  anti-PD-1 10-gene set. The top enriched pathway is **"Constitutive
  Signalling by Aberrant PI3K in Cancer"** at **FDR 1.2e-7**.

- **`mechanism_umap_final.png`** and **`mechanism_umap_final.json`** — the
  mechanism-of-action UMAP projection (figure plus the underlying coordinates
  and labels).

- **`INTERPRETABILITY_REPORT.md`** — the written interpretability report
  summarizing these analyses.

Additional supporting caches and intermediate query files are also included in
`artifacts/` for traceability.

## References for the external resources

- Open Targets Platform: Ochoa et al. (2023), *Nucleic Acids Research* 51:D1353.
- Reactome: Milacic et al. (2024), *Nucleic Acids Research* 52:D672.
- MyGene.info: Xin et al. (2016), *Genome Biology* 17:91.
