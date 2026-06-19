# ESM-2 Entity Initialization

This directory contains the ESM-2 protein-language-model embeddings used to
initialize the entity embeddings of the monoclonal-antibody (mAb) entities in
the knowledge graph, together with the metadata describing exactly how that
initialization was performed.

## Layout

This directory mirrors the path contract expected by the training scripts, so
`--biologics-dir esm_init` (the default) resolves the embeddings automatically.

- **`data/mab_inn_sequences_split.csv`** — the antibody amino-acid sequences
  (heavy and light chains) used as input to ESM-2, keyed by mAb / INN.

- **`artifacts/kg_vocab_onco_filtered_ESM_640/entity_embeddings.npy`** — the
  per-mAb **source** ESM-2 embeddings. For each antibody the heavy-chain and
  light-chain ESM-2 representations are concatenated to a single
  **1280-dimensional** vector per mAb.

- **`artifacts/kg_vocab_onco_filtered_ESM_640/entities.csv`** — the entity index
  (entity → row) giving the row order of the source embedding array.

- **`entity_init.npy`** — the **640-dimensional additive init** actually used
  to initialize entity embeddings. It is formed from the 1280-d source vector
  by splitting it into two halves and adding them elementwise:
  `add(first_half, second_half)` → 640-d. This is the array consumed at
  training time to seed mAb entity embeddings.

- **`entity_init_metadata.json`** — records how the init was built and which
  entities were covered. In particular it records that **483 of 684 mAb
  entities were initialized** from ESM-2; the remaining mAb entities (those
  lacking an INN link or a usable sequence) were left at the model's default
  random initialization, `uniform(-0.01, 0.01)`. It also records the source
  dimension (1280), the init dimension (640), and the fusion rule
  (`add(first_half, second_half)`).

## Regenerating the ESM-2 source embeddings

The provided `.npy` arrays are pre-computed, so they can be used directly. To
regenerate the source embeddings from scratch, run ESM-2 over the heavy and
light chains in `mab_inn_sequences_split.csv` and concatenate the per-chain
representations. This requires the ESM-2 model weights and either the
[`fair-esm`](https://github.com/facebookresearch/esm) package or Hugging Face
[`transformers`](https://github.com/huggingface/transformers) (see
`requirements.txt`). The 640-d `entity_init.npy` is then obtained by splitting
each 1280-d vector in half and adding the two halves.

ESM-2 reference: Lin et al. (2023), "Evolutionary-scale prediction of
atomic-level protein structure with a language model", *Science* 379:1123.
doi:[10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)
