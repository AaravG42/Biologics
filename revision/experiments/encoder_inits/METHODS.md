# Encoder-Init Ablation: Antibody-Sequence Embeddings

Reviewer-requested ablation comparing a **general** protein language model
(ESM-2) against **antibody-specific** encoders (AbLang, IgBert) as sources of
mAb-entity initialization vectors. All inference is **CPU-only**
(`CUDA_VISIBLE_DEVICES=""`).

## Inputs (read-only)

- Antibody sequences: `/home/aarav/Biologics/esm_init/data/mab_inn_sequences_split.csv`
  (`inn_number`, `heavy_chain_sequence`, `light_chain_sequence`).
- KG triples: `/home/aarav/KGEmb/data/IMGT_no_leakage_directional_clinical/{train,valid,test}`.

## mAb -> INN mapping

Mirrors `utils/imgt_clinical.py` (`extract_mab_to_inn_mapping`,
`MAB_PATTERN`, `INN_PATTERN`, `STRUCTURE_LINK_RELATION`):

- `MAB_PATTERN = imgt:mAb_(\d+)$`, `INN_PATTERN = imgt:(\d+)$`,
  relation `imgt:isLinkedToStructureAccessNumb`.
- For every triple `(h, r, t)` with `r == imgt:isLinkedToStructureAccessNumb`
  and `h` matching `imgt:mAb_<N>` and `t` matching `imgt:<INN>`, map
  `imgt:mAb_<N> -> <INN>`. All such triples live in the `train` split (666).

## mAb selection

A mAb entity is embedded **iff** it has (a) a resolvable mAb-entity <-> INN
mapping AND (b) BOTH a non-empty heavy and a non-empty light chain. mAbs with an
empty heavy or light chain are skipped. Records are ordered by numeric mAb id
for determinism. (See `encoder_summary.json` for the exact counts and the list
of skipped mAbs.)

## Encoders

Each encoder embeds the heavy and light chains **separately**, mean-pools the
per-residue embeddings within each chain, then **concatenates** heavy ⊕ light to
form one vector per mAb.

| Encoder | Model | Package | Per-chain dim | Concat dim | Type |
|---------|-------|---------|---------------|------------|------|
| esm2    | `facebook/esm2_t30_150M_UR50D` | transformers (AutoTokenizer/AutoModel) | 640 | 1280 | general PLM |
| ablang  | AbLang-1 `heavy` + `light` pretrained | ablang | 768 | 1536 | antibody-specific |
| igbert  | `Exscientia/IgBert` | transformers (BertModel/BertTokenizer) | 1024 | 2048 | antibody-specific |

Pooling details (identical pooling concept — mean over residue token embeddings):

- **ESM-2**: tokenize each chain, take `last_hidden_state`, drop the CLS and EOS
  special tokens, mean-pool the remaining residue tokens. Concat heavy ⊕ light.
- **AbLang**: heavy model for the heavy chain, light model for the light chain;
  `mode='rescoding'` per-residue embeddings (AbLang-1 already excludes its
  start/end tokens), mean-pooled per chain, then concat.
- **IgBert**: paired input `"<spaced heavy> [SEP] <spaced light>"` with the
  model's CLS/SEP special tokens (per the model card, residues are
  space-separated and chains paired). The `last_hidden_state` is split at the
  first `[SEP]`; heavy residue tokens (between CLS and first SEP) and light
  residue tokens (between first and last SEP) are mean-pooled separately, then
  concatenated.

### Sequence-length handling (per-encoder architectural limits)

The CSV stores **full-length** chains (heavy median ~449, max 1230; light median
~214) — i.e. variable + constant regions. Encoders are fed the full chain up to
their own architectural position limit; truncation is N-terminal (the variable
Fv domain is N-terminal) and recorded in `encoder_summary.json`:

- **ESM-2** (absolute positions, `max_position_embeddings=1026`): chains capped
  at 1024 residues. This affects exactly 1 chain in the dataset (the single
  1230-residue heavy chain).
- **IgBert** (`max_position_embeddings=40000`): no truncation needed.
- **AbLang-1**: models antibody **variable domains** only — a hard positional
  limit of 157 residues. Full chains are truncated to their N-terminal 157
  residues (the Fv region AbLang is designed for). This is a property of
  AbLang-1, not an arbitrary choice; the count of truncated chains is recorded.

## Fair reduction (identical for every successful encoder)

Applied to the `[N, D]` matrix of per-mAb concatenated vectors:

1. **Standardize** features to zero-mean / unit-variance across the mAb set
   (`sklearn.preprocessing.StandardScaler`).
2. **PCA toward 640 components** (`sklearn.decomposition.PCA`,
   `random_state=42`). The *feature* dim of every encoder is `>= 640`
   (1280 / 1536 / 2048); if a concat feature dim were `< 640` the pipeline
   raises rather than padding features.
   **Sample-count constraint (forced deviation):** PCA can emit at most
   `min(n_samples, n_features)` components. With only **483 mAbs**
   (`n_samples = 483 < 640`), a literal 640-component PCA is mathematically
   impossible (sklearn raises `n_components must be between 0 and 483`). We
   therefore fit PCA with `n_components = min(640, n_samples, n_features) = 483`
   and **zero-pad the trailing 157 columns** up to 640. Those padded columns are
   exactly the directions PCA has no data to populate (the data null space), so
   no signal is fabricated; the padding only conforms the output to the required
   `[N, 640]` shape. This is applied **identically** to every encoder
   (each: 483 components fit, 157 zero-padded).
3. **L2-normalize** each 640-d mAb vector (after padding).
4. **Scale by 0.01** — matching the model's random-init scale
   `uniform(-0.01, 0.01)`. (Verified: every output row has L2 norm exactly
   0.01 and `max|value| < 0.01`.)

## Outputs

- `<enc>_mab_vectors.npz` for each successful encoder, with arrays:
  - `names`: mAb entity strings (e.g. `"imgt:mAb_1365"`),
  - `vecs`: `float32`, shape `[num_mab, 640]`, row-aligned to `names`.
- `encoder_summary.json`: per-encoder run status, package/model version, raw
  concat dim, num mAbs embedded, PCA explained-variance (sum + per-component for
  the first 640), truncation info, plus the global selection criteria and
  skipped-mAb lists.
- `METHODS.md`: this file.

## Reproduction

```bash
cd /home/aarav/KGEmb/revision/experiments/encoder_inits
CUDA_VISIBLE_DEVICES="" python3 run_encoders.py
```

`common.py` holds the mapping/selection and fair-reduction logic;
`run_encoders.py` holds the three encoder functions and the driver.

## Outcome

All three encoders **succeeded** (`ran: true` in `encoder_summary.json`):

| Encoder | Package version | Raw concat dim | mAbs embedded | PCA fit / zero-padded | Output |
|---------|-----------------|----------------|---------------|------------------------|--------|
| esm2    | transformers 4.57.6 | 1280 | 483 | 483 / 157 | `esm2_mab_vectors.npz` |
| ablang  | ablang 0.3.1 (AbLang-1) | 1536 | 483 | 483 / 157 | `ablang_mab_vectors.npz` |
| igbert  | transformers 4.57.6 | 2048 | 483 | 483 / 157 | `igbert_mab_vectors.npz` |

- **mAb set**: 483 mAbs embedded; 89 mAbs skipped for an empty heavy or light
  chain; 0 missing sequence rows. 572 total mAb->INN structure links.
- **ESM-2 truncation**: exactly 1 chain truncated (`imgt:mAb_1248:H`, the single
  1230-residue heavy chain, capped at 1024).
- **AbLang truncation caveat**: 961 of 966 chains were truncated to the
  N-terminal 157 residues, because AbLang-1 only accepts variable-domain-length
  inputs while the CSV stores full-length (variable + constant) chains. This is
  the principal fairness caveat for interpreting the AbLang arm: AbLang sees
  essentially only the Fv region, whereas ESM-2 and IgBert see the full chains.

See `encoder_summary.json` for full per-encoder metadata, the complete
explained-variance vectors, and the list of skipped mAbs.
