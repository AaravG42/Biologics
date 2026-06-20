"""Generate antibody-sequence embeddings for the encoder-init ablation.

Runs ESM-2 (general PLM) vs antibody-specific encoders (AbLang, IgBert) on CPU.
For each encoder: mean-pool residue embeddings per chain, concat heavy+light,
then apply the identical fair-reduction pipeline (standardize -> PCA-640 ->
L2-norm -> 0.01 scale) and save <enc>_mab_vectors.npz.

CPU ONLY. Run with CUDA_VISIBLE_DEVICES="".
"""

from __future__ import annotations

import json
import os
import platform
import traceback
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, List

import numpy as np

import common

OUT_DIR = Path("/home/aarav/KGEmb/revision/experiments/encoder_inits")


def _pkg_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# ESM-2
# ---------------------------------------------------------------------------
def embed_esm2(records: List[Dict[str, str]]) -> Dict[str, object]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_id = "facebook/esm2_t30_150M_UR50D"
    # ESM-2 uses absolute position embeddings (max_position_embeddings=1026),
    # so residue tokens must be <= 1024 (+CLS+EOS). Truncate the (single) longer
    # heavy chain at the N-terminus; record any truncations.
    max_residues = 1024
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to("cpu")
    torch.set_grad_enabled(False)
    truncated = []

    def mean_pool_chain(seq: str, name: str, chain: str) -> np.ndarray:
        if len(seq) > max_residues:
            seq = seq[:max_residues]
            truncated.append(f"{name}:{chain}")
        enc = tok([seq], return_tensors="pt", padding=False, truncation=False)
        out = model(**enc).last_hidden_state[0]  # [L, 640] incl CLS/EOS
        # Drop CLS (first) and EOS (last) special tokens -> residue tokens only.
        residues = out[1:-1]
        return residues.mean(dim=0).numpy()

    heavy_dim = model.config.hidden_size
    rows = []
    for i, r in enumerate(records):
        h = mean_pool_chain(r["heavy"], r["name"], "H")
        l = mean_pool_chain(r["light"], r["name"], "L")
        rows.append(np.concatenate([h, l]))
        if (i + 1) % 50 == 0:
            print(f"  esm2 {i + 1}/{len(records)}", flush=True)
    concat = np.vstack(rows)
    return {
        "concat": concat,
        "model": model_id,
        "package": "transformers",
        "package_version": _pkg_version("transformers"),
        "per_chain_dim": int(heavy_dim),
        "pooling": "mean over residue tokens (CLS/EOS excluded), heavy(+)light concat",
        "max_residues_per_chain": max_residues,
        "truncated_chains": truncated,
    }


# ---------------------------------------------------------------------------
# AbLang (AbLang-1)
# ---------------------------------------------------------------------------
def embed_ablang(records: List[Dict[str, str]]) -> Dict[str, object]:
    import ablang

    heavy_model = ablang.pretrained("heavy", device="cpu")
    heavy_model.freeze()
    light_model = ablang.pretrained("light", device="cpu")
    light_model.freeze()

    # AbLang-1 models antibody VARIABLE domains only: hard positional limit of
    # 157 residues (160 positions incl. start/end). Full-length chains in the
    # CSV include constant regions, so we truncate to the N-terminal 157
    # residues (the variable domain is N-terminal -> this captures the Fv that
    # AbLang is designed for). Truncations are recorded.
    max_residues = 157
    truncated = []

    def mean_pool(model, seq: str, name: str, chain: str) -> np.ndarray:
        if len(seq) > max_residues:
            seq = seq[:max_residues]
            truncated.append(f"{name}:{chain}")
        # rescoding returns per-residue embeddings (already excludes start/end
        # special tokens in AbLang-1), length == len(seq), dim 768.
        rc = model([seq], mode="rescoding")
        arr = np.asarray(rc[0])  # [L, 768]
        return arr.mean(axis=0)

    rows = []
    per_chain_dim = None
    for i, r in enumerate(records):
        h = mean_pool(heavy_model, r["heavy"], r["name"], "H")
        l = mean_pool(light_model, r["light"], r["name"], "L")
        if per_chain_dim is None:
            per_chain_dim = int(h.shape[0])
        rows.append(np.concatenate([h, l]))
        if (i + 1) % 50 == 0:
            print(f"  ablang {i + 1}/{len(records)}", flush=True)
    concat = np.vstack(rows)
    return {
        "concat": concat,
        "model": "ablang heavy + ablang light (AbLang-1 pretrained)",
        "package": "ablang",
        "package_version": _pkg_version("ablang"),
        "per_chain_dim": per_chain_dim,
        "pooling": "mean over rescoding residue embeddings, heavy(+)light concat",
        "max_residues_per_chain": max_residues,
        "truncated_chains_count": len(truncated),
        "note": (
            "AbLang-1 handles only variable-domain length (<=157 residues); "
            "full chains truncated to N-terminal 157 (the Fv region)."
        ),
    }


# ---------------------------------------------------------------------------
# IgBert
# ---------------------------------------------------------------------------
def embed_igbert(records: List[Dict[str, str]]) -> Dict[str, object]:
    import torch
    from transformers import BertModel, BertTokenizer

    model_id = "Exscientia/IgBert"
    tok = BertTokenizer.from_pretrained(model_id)
    model = BertModel.from_pretrained(model_id).eval().to("cpu")
    torch.set_grad_enabled(False)

    def spaced(seq: str) -> str:
        return " ".join(list(seq))

    def embed_pair(heavy: str, light: str) -> np.ndarray:
        # IgBert paired input: "H H H [SEP] L L L" with CLS/SEP added.
        paired = [spaced(heavy) + " [SEP] " + spaced(light)]
        toks = tok.batch_encode_plus(
            paired, add_special_tokens=True, return_tensors="pt", padding=False
        )
        ids = toks["input_ids"][0]
        hidden = model(
            input_ids=toks["input_ids"], attention_mask=toks["attention_mask"]
        ).last_hidden_state[0]  # [T, 1024]

        # Token layout: [CLS] heavy... [SEP] light... [SEP]
        cls_id = tok.cls_token_id
        sep_id = tok.sep_token_id
        sep_positions = (ids == sep_id).nonzero(as_tuple=True)[0].tolist()
        first_sep = sep_positions[0]
        # heavy residues: indices 1 .. first_sep-1 ; light: first_sep+1 .. second_sep-1
        heavy_h = hidden[1:first_sep].mean(dim=0).numpy()
        light_h = hidden[first_sep + 1 : sep_positions[-1]].mean(dim=0).numpy()
        return np.concatenate([heavy_h, light_h])

    rows = []
    for i, r in enumerate(records):
        rows.append(embed_pair(r["heavy"], r["light"]))
        if (i + 1) % 50 == 0:
            print(f"  igbert {i + 1}/{len(records)}", flush=True)
    concat = np.vstack(rows)
    return {
        "concat": concat,
        "model": model_id,
        "package": "transformers",
        "package_version": _pkg_version("transformers"),
        "per_chain_dim": int(model.config.hidden_size),
        "pooling": "mean over per-chain residue tokens of paired H[SEP]L input, heavy(+)light concat",
    }


ENCODERS = {
    "esm2": embed_esm2,
    "ablang": embed_ablang,
    "igbert": embed_igbert,
}


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", (
        "Must run with CUDA_VISIBLE_DEVICES='' (CPU only)."
    )
    records, map_info = common.build_mab_records()
    names = np.array([r["name"] for r in records], dtype=object)
    print(f"Qualifying mAbs: {len(records)}", flush=True)

    summary: Dict[str, object] = {
        "num_qualifying_mabs": len(records),
        "selection_criteria": (
            "mAb has resolvable imgt:mAb_<N> -> imgt:<INN> via "
            "'imgt:isLinkedToStructureAccessNumb' AND both heavy & light "
            "sequences non-empty"
        ),
        "mab_to_inn_links_total": map_info["num_mab_inn_links"],
        "skipped_empty_chain_count": len(map_info["skipped_empty_chain"]),
        "skipped_empty_chain": map_info["skipped_empty_chain"],
        "skipped_no_sequence_row_count": len(map_info["skipped_no_sequence_row"]),
        "skipped_no_sequence_row": map_info["skipped_no_sequence_row"],
        "fair_reduction": {
            "standardize": "StandardScaler zero-mean/unit-var across mAb set",
            "pca_target_components": common.PCA_DIM,
            "pca_random_state": common.RANDOM_STATE,
            "l2_normalize": True,
            "scale": common.INIT_SCALE,
            "note": (
                "PCA cannot emit more than min(n_samples, n_features) "
                "components. With 483 mAbs < 640, PCA is fit with "
                "min(640, n_samples, n_features) components and the trailing "
                "columns are zero-padded up to 640 (the PCA null space) before "
                "L2-norm and 0.01 scaling. Applied identically to every encoder."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "cpu_only": True,
            "numpy": np.__version__,
            "sklearn": _pkg_version("scikit-learn"),
            "torch": _pkg_version("torch"),
        },
        "encoders": {},
    }

    for enc_name, fn in ENCODERS.items():
        print(f"\n=== Encoder: {enc_name} ===", flush=True)
        entry: Dict[str, object] = {"ran": False}
        try:
            # Cache raw concat matrices so PCA/reduction can be re-run without
            # repeating CPU encoder inference.
            cache_path = OUT_DIR / f"_concat_{enc_name}.npz"
            if cache_path.is_file():
                cached = np.load(cache_path, allow_pickle=True)
                result = {k: cached[k].item() for k in cached.files if k != "concat"}
                result["concat"] = cached["concat"]
                print(f"  loaded cached concat {result['concat'].shape}", flush=True)
            else:
                result = fn(records)
                meta = {k: v for k, v in result.items() if k != "concat"}
                np.savez(cache_path, concat=result["concat"], **{k: np.array(v, dtype=object) for k, v in meta.items()})
            concat = result["concat"]
            print(f"  raw concat shape: {concat.shape}", flush=True)
            vecs, evr, pca_info = common.fair_reduction(concat)
            out_path = OUT_DIR / f"{enc_name}_mab_vectors.npz"
            np.savez(out_path, names=names, vecs=vecs.astype(np.float32))
            entry.update(
                {
                    "ran": True,
                    "model": result["model"],
                    "package": result["package"],
                    "package_version": result["package_version"],
                    "per_chain_dim": result["per_chain_dim"],
                    "raw_concat_dim": int(concat.shape[1]),
                    "num_mabs_embedded": int(vecs.shape[0]),
                    "pooling": result["pooling"],
                    "pca_components_fit": pca_info["pca_components_fit"],
                    "pca_components_zero_padded": pca_info["pca_components_zero_padded"],
                    "pca_explained_variance_fit_sum": float(np.sum(evr)),
                    "pca_explained_variance_ratio": evr,
                    "output_file": str(out_path),
                }
            )
            for opt_key in (
                "max_residues_per_chain",
                "truncated_chains",
                "truncated_chains_count",
                "note",
            ):
                if opt_key in result:
                    entry[opt_key] = result[opt_key]
            print(
                f"  OK -> {out_path} | concat_dim={concat.shape[1]} "
                f"| pca_fit={pca_info['pca_components_fit']} "
                f"padded={pca_info['pca_components_zero_padded']} "
                f"| EVR(fit)={np.sum(evr):.4f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            entry.update(
                {
                    "ran": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
        summary["encoders"][enc_name] = entry

    (OUT_DIR / "encoder_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nWrote encoder_summary.json", flush=True)


if __name__ == "__main__":
    main()
