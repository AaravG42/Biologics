# Knowledge-Graph–Driven Repurposing of Therapeutic Monoclonal Antibodies in Oncology

Code, data, and result artifacts for the manuscript:

> **Repurposing Therapeutic Monoclonal Antibodies in Oncology Using a
> Pharmacologically Validated Knowledge Graph–Based Framework.**
> Aarav Gupta, Anirban Pal, Archana Kumari Redhu, Angel Singhvi, Aryan Shah,
> Debabrata Maiti, Vikram Gota. Submitted to the *British Journal of Clinical
> Pharmacology* (2026).

This repository accompanies the manuscript's *Code and Data Availability*
statement. It contains the knowledge-graph embedding pipeline, the
leakage-controlled datasets, the trained-model records, and the biological
validation artifacts needed to reproduce the reported experiments. A complete,
reviewer-oriented reconciliation of **every reported number to a committed
artifact** is in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## What this study does (and how it relates to prior work)

The study takes an oncology knowledge graph of therapeutic monoclonal antibodies
and asks how well knowledge-graph embedding (KGE) models can predict clinical
indications — i.e. propose repurposing candidates — once a specific data-leakage
artifact is removed.

**Built on prior work (credited, not claimed as new):** the dataset is the
**IMGT/mAbOnco-KG** of *Sanou et al.* (refs below), and the general task of KGE
on this resource, the link-prediction benchmark setting, and the
embedding-visualization style of analysis originate with that work. See
[`docs/DATA.md`](docs/DATA.md).

**This work's specific additions:**
1. A **reciprocal-leakage audit** of the original splits (the structurally
   equivalent inverse `isClinicalIndicationOf` triples leak `hasClinicalIndication`
   test answers into training) and a leakage-controlled re-split.
2. Evaluation of **Euclidean and hyperbolic** KGE models (RotE, TransE, RefE,
   MurE, AttE, CP, RotH, RefH, AttH) on the cleaned, forward-only clinical task.
3. **ESM-2 protein-language-model initialization** of antibody entity
   embeddings from heavy/light-chain sequences.
4. A **pharmacological plausibility** read-out of the top predictions against the
   Open Targets Platform and Reactome (five case studies).

The honest headline is the **leakage audit**: once reciprocal leakage is removed,
performance is modest (best test MRR ≈ 0.32, median rank 7), which is a more
realistic estimate of repurposing-KGE capability than leaked benchmarks suggest.
Known limitations and items to correct in any resubmission are listed in
[`docs/PROVENANCE.md` §6](docs/PROVENANCE.md).

## Repository layout

```
.
├── README.md                 # this file
├── docs/
│   ├── PROVENANCE.md          # reconciles every reported number to an artifact
│   ├── REPRODUCE.md           # end-to-end reproduction commands
│   └── DATA.md                # dataset provenance, licensing (Sanou et al. / IMGT / EUPL)
├── models/ optimizers/ datasets/ utils/ run.py   # KGE library (derived from HazyResearch/KGEmb, Apache-2.0)
├── scripts/                   # dataset construction, HPO, eval, repurposing inference, interpretability
├── clinical_eval.py           # clinical-indication evaluation of a checkpoint
├── data/                      # IMGT, IMGT_no_leakage, IMGT_no_leakage_directional_clinical
├── esm_init/                  # ESM-2 source + 640-d additive entity init + metadata
├── results/
│   ├── checkpoints/           # per-model selected trial (config + metrics; RotE incl. model.pt)
│   ├── hpo_all_trials.csv     # full per-trial HPO record
│   ├── summary.json           # HPO summary
│   └── tables/                # tables regenerated from the artifacts
└── interpretability/
    ├── README.md
    └── artifacts/             # top-k predictions, Open Targets / Reactome validation, UMAP
```

## Quickstart

```bash
pip install -r requirements.txt
source set_env.sh

# Reproduce the clinical-indication metrics of the best model (no training needed)
python clinical_eval.py results/checkpoints/RotE_trial_101 --batch-size 256

# Regenerate the benchmark table from committed artifacts
cat results/tables/table_s3_reproduced.csv
```

Full pipeline (data construction → ESM init → training → evaluation →
interpretability) is in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Data, licensing, and attribution

- **Code** in this repository is licensed under **Apache-2.0** ([`LICENSE`](LICENSE)).
  The KGE library files (`models/`, `optimizers/`, `datasets/`, `utils/`,
  `run.py`) are **derived from** [HazyResearch/KGEmb](https://github.com/HazyResearch/KGEmb)
  (Chami et al., *Low-Dimensional Hyperbolic Knowledge Graph Embeddings*, ACL
  2020), Apache-2.0; see [`NOTICE`](NOTICE).
- **The dataset is not our work.** It is the IMGT/mAbOnco-KG of Sanou et al.,
  derived from IMGT resources; it carries its own terms (EUPL-1.2 source
  repository; IMGT data terms). See [`docs/DATA.md`](docs/DATA.md). Please cite
  Sanou et al. if you use the data.
- Biological validation uses the public Open Targets, Reactome, and MyGene.info
  services; please cite them.

### Key references

- Sanou, G., Manso, T., Todorov, K., Giudicelli, V., Duroux, P. (2026).
  Therapeutic monoclonal antibodies repurposing in oncology via IMGT/mAb-KG
  embeddings. *BMC Medical Informatics and Decision Making* 26:89.
  doi:10.1186/s12911-026-03394-4
- Sanou, G., Manso, T., Todorov, K., Giudicelli, V., Duroux, P., Kossida, S.
  (2024). IMGT/mAb-KG: The knowledge graph for therapeutic monoclonal
  antibodies. *Frontiers in Immunology* 15:1393839.
  doi:10.3389/fimmu.2024.1393839
- Chami, I., et al. (2020). Low-Dimensional Hyperbolic Knowledge Graph
  Embeddings. *ACL 2020*.
- Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein
  structure with a language model (ESM-2). *Science* 379:1123.

See [`CITATION.cff`](CITATION.cff) for citation metadata.
