# Reproduction Guide

This guide reproduces the experiments in the manuscript end-to-end. All commands
are run from the repository root unless stated otherwise. The committed data,
ESM initialization, checkpoints, and result tables let you verify most outputs
**without** any training; the training commands regenerate them from scratch.

## 0. Environment

```bash
pip install -r requirements.txt
source set_env.sh          # sets PYTHONPATH, DATA_PATH=$(pwd)/data, LOG_DIR
```

Training/eval use a CUDA GPU by default (`--device cuda`/`auto`); CPU works via
`--device cpu`. The original experiments ran on a single NVIDIA H200.

## 1. Datasets (committed; rebuild optional)

The three datasets are committed under `data/` (see [`DATA.md`](DATA.md) for
provenance). `data/IMGT` is Sanou et al.'s processed IMGT/mAbOnco-KG split. The
leakage-controlled variants are rebuilt from it with:

```bash
# Remove reciprocal (inverse-relation) cross-split leakage
python scripts/create_imgt_no_leakage.py \
    --source-dir data/IMGT --output-dir data/IMGT_no_leakage

# Drop the inverse imgt:isClinicalIndicationOf relation (forward-only clinical)
python scripts/create_imgt_directional_clinical.py \
    --source-dir data/IMGT_no_leakage \
    --output-dir data/IMGT_no_leakage_directional_clinical \
    --drop-relation imgt:isClinicalIndicationOf
```

Each writes `{train,valid,test}` + `{train,valid,test}.pickle` + `to_skip.pickle`
and reports the reciprocal-leakage counts it removes (manuscript Table 2 / S2).

## 2. ESM-2 entity initialization (committed; rebuild optional)

The 640-d additive init is committed at `esm_init/entity_init.npy` (built from
`esm_init/artifacts/kg_vocab_onco_filtered_ESM_640/entity_embeddings.npy`). The
training scripts build it on demand from `--biologics-dir` (default `esm_init/`).
To regenerate the 1,280-d ESM-2 source from the antibody sequences, see
[`../esm_init/README.md`](../esm_init/README.md) (requires `fair-esm` /
`transformers`).

## 3. Train the 9-model benchmark (manuscript Table 3 / S3)

Full hyperparameter sweep over all models on the leakage-free, directional
dataset (rank 640, ESM additive init applied automatically):

```bash
python scripts/hpo_imgt_clinical_directional.py \
    --rank 640 --trials-per-model 16 \
    --output-dir runs/clinical_hpo --device auto --gpu-ids 0
```

This is expensive (each trial trains up to 200 epochs). To reproduce a **single**
model at the configuration reported in the paper, train with that model's
committed config, e.g. for RotE:

```bash
# config from results/checkpoints/RotE_trial_101/config.json
python scripts/hpo_imgt_clinical_directional.py --models RotE \
    --rank 640 --trials-per-model 1 --device auto --gpu-ids 0
```

Model selection (per the manuscript) is by **lowest validation forward-clinical
tail Mean Rank**. The trial backing each Table S3 row is listed in
[`PROVENANCE.md`](PROVENANCE.md §3). `results/hpo_all_trials.csv` contains the
full per-trial record; `results/tables/table_s3_reproduced.csv` is regenerated
directly from the committed `metrics.json` files.

> Note the RotE checkpoint caveat in [`PROVENANCE.md`](PROVENANCE.md §3): the
> manuscript's RotE row is `metrics.before_hardneg.json`; the committed
> `model.pt` is the later hard-negative state.

## 4. Evaluate a checkpoint on the clinical-indication relation

```bash
python clinical_eval.py results/checkpoints/RotE_trial_101 --batch-size 256
```

Prints forward (mAb→disease), reverse (disease→mAb), and combined MR/MRR/Hits@k
on valid and test. Add `--head-csv-out <path>` for the per-disease head-prediction
breakdown.

## 5. Repurposing inference (disease → ranked mAb candidates)

```bash
python scripts/reverse_clinical_mab_candidates.py \
    --trial-dir results/checkpoints/RotE_trial_101 \
    --disease 'imgt:Chronic_lymphocytic_leukemia_(CLL)' \
    --top-k-mabs 15 --top-k-study-products 40
```

## 6. Interpretability & biological validation

```bash
# Per-query top-K predictions (drives the case studies)
python scripts/atth_interpret.py \
    --trial_dir results/checkpoints/RotE_trial_101 --top_k 50

# Mechanism-of-action UMAP over the 297 forward queries
python scripts/mechanism_cluster.py
```

The Open Targets / Reactome / MyGene biological verification (Tables S6–S11) was
performed with the external **celltype-agent** tool against public APIs; the
committed outputs are an April-2026 snapshot under
`interpretability/artifacts/`. See
[`../interpretability/README.md`](../interpretability/README.md) for the exact
tools/endpoints and the planned systematic, blinded re-scoring.

## 7. Knowledge-graph structure figures (Table 1 / Figures 1–2)

```bash
python scripts/analyze_imgt_graph.py        # degree stats, Gromov delta-hyperbolicity
python scripts/analyze_collective_kg.py     # collective KG structural figure
```

## Reproducibility notes

- Seeds are deterministic: per-trial torch seed `= 42 + 1000*(model_index+1) + trial_index`.
- External databases (Open Targets, Reactome, MyGene) are **live**; pin their
  release versions when exact reproduction of the biological tags is required.
- A complete, reconciled mapping of every reported number to its artifact is in
  [`PROVENANCE.md`](PROVENANCE.md).
