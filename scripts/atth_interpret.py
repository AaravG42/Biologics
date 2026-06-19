"""Inference + top-K dump for AttH on hasClinicalIndication forward queries.

Outputs per-query ranks and top-K predicted entities (entity names), plus
ground-truth tail, into analysis_outputs/.../AttH/trial_000/interpret_topk.json
"""
import argparse
import json
import os
import sys
import pickle
from types import SimpleNamespace

import numpy as np
import torch

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("DATA_PATH", str(REPO_ROOT / "data"))

import models
from datasets.kg_dataset import KGDataset
from datasets.process import get_idx


def main(trial_dir: str, top_k: int = 50):
    with open(os.path.join(trial_dir, "config.json")) as f:
        cfg = json.load(f)
    args = SimpleNamespace(**cfg)

    data_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(data_path, False)
    args.sizes = dataset.get_shape()

    ent2idx, rel2idx = get_idx(data_path)
    id2ent = {i: e for e, i in ent2idx.items()}
    id2rel = {i: r for r, i in rel2idx.items()}

    model = getattr(models, args.model)(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    state = torch.load(os.path.join(trial_dir, "model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    filters = dataset.get_filters()  # {'lhs': dict, 'rhs': dict}
    rhs_filter = filters["rhs"]

    REL_ID = 6  # imgt:hasClinicalIndication
    test = dataset.get_examples("test").numpy()
    fwd = test[test[:, 1] == REL_ID]
    print(f"forward test queries (rel={id2rel[REL_ID]}): {len(fwd)}")

    results = []
    with torch.no_grad():
        cand_rhs, _ = model.get_rhs(
            torch.zeros((1, 3), dtype=torch.long, device=device), eval_mode=True
        )  # (n_ent, d)
        # per-query scoring
        batch = torch.from_numpy(fwd).to(device)
        q_emb, q_bias = model.get_queries(batch)
        rhs_bias = model.bt.weight
        # score: (n_q, n_ent)
        scores = model.score((q_emb, q_bias), (cand_rhs, rhs_bias), eval_mode=True)
        scores_np = scores.detach().cpu().numpy()

        # apply filter: mask out other known true tails for (h, r) except the ground truth
        for i, (h, r, t) in enumerate(fwd):
            others = list(rhs_filter.get((int(h), int(r)), []))
            # keep the ground-truth tail unmasked
            for o in others:
                if o != t:
                    scores_np[i, o] = -1e9
        # compute rank of true tail
        true_scores = scores_np[np.arange(len(fwd)), fwd[:, 2]]
        # number of entities with score >= true_score (standard-competition rank)
        ranks = (scores_np >= true_scores[:, None]).sum(axis=1)

        # top-K predicted entity ids (descending by score)
        topk_idx = np.argsort(-scores_np, axis=1)[:, :top_k]

    for i, (h, r, t) in enumerate(fwd):
        rank = int(ranks[i])
        row = {
            "query_idx": int(i),
            "head_id": int(h),
            "head": id2ent[int(h)],
            "relation": id2rel[int(r)],
            "tail_id": int(t),
            "tail_true": id2ent[int(t)],
            "rank": rank,
            "top_k": [
                {
                    "pos": k + 1,
                    "ent_id": int(topk_idx[i, k]),
                    "ent": id2ent[int(topk_idx[i, k])],
                    "score": float(scores_np[i, int(topk_idx[i, k])]),
                }
                for k in range(top_k)
            ],
        }
        results.append(row)

    out_path = os.path.join(trial_dir, f"interpret_topk_{top_k}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "trial_dir": trial_dir,
                "dataset": args.dataset,
                "relation": id2rel[REL_ID],
                "n_queries": len(fwd),
                "mean_rank": float(ranks.mean()),
                "median_rank": float(np.median(ranks)),
                "hits@1": float((ranks == 1).mean()),
                "hits@10": float((ranks <= 10).mean()),
                "mrr": float((1.0 / ranks).mean()),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"wrote {out_path}")
    print(
        f"MR={ranks.mean():.2f}  median={np.median(ranks):.1f}  "
        f"MRR={(1.0/ranks).mean():.3f}  H@1={(ranks==1).mean():.3f}  "
        f"H@10={(ranks<=10).mean():.3f}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trial_dir",
        default=str(REPO_ROOT / "results" / "checkpoints" / "AttH_trial_000"),
    )
    p.add_argument("--top_k", type=int, default=50)
    a = p.parse_args()
    main(a.trial_dir, a.top_k)
