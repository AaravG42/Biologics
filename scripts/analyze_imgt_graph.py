"""Analyze structural properties of an IMGT knowledge graph subset.

This script computes:
1. Degree distribution statistics and plots.
2. Approximate or exact Gromov delta-hyperbolicity on the largest component.

The graph is treated as an undirected simple graph for structural analysis.
"""

import argparse
import csv
import itertools
import json
import math
import os
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze IMGT graph structure")
    parser.add_argument(
        "--dataset",
        default="IMGT",
        help="Dataset directory name under DATA_PATH",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override DATA_PATH root directory",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/IMGT",
        help="Directory where reports and plots will be written",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Raw split files to load",
    )
    parser.add_argument(
        "--exact-component-limit",
        type=int,
        default=96,
        help="Use exact hyperbolicity if largest component size is at most this value",
    )
    parser.add_argument(
        "--delta-sample-nodes",
        type=int,
        default=96,
        help="Number of nodes to sample from the largest component for approximate delta",
    )
    parser.add_argument(
        "--delta-sample-quartets",
        type=int,
        default=200000,
        help="Number of random quartets to evaluate for approximate delta",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible sampling",
    )
    return parser.parse_args()


def resolve_dataset_dir(args):
    data_root = args.data_path or os.environ.get("DATA_PATH") or os.path.join(os.getcwd(), "data")
    dataset_dir = Path(data_root) / args.dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    return dataset_dir


def load_graph(dataset_dir, splits):
    adjacency = defaultdict(set)
    directed_triples = 0
    relation_counts = Counter()
    self_loops = 0

    for split in splits:
        split_path = dataset_dir / split
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        with open(split_path, "r") as handle:
            for line in handle:
                head, relation, tail = line.rstrip("\n").split("\t")
                directed_triples += 1
                relation_counts[relation] += 1
                adjacency.setdefault(head, set())
                adjacency.setdefault(tail, set())
                if head == tail:
                    self_loops += 1
                    continue
                adjacency[head].add(tail)
                adjacency[tail].add(head)

    return adjacency, directed_triples, relation_counts, self_loops


def connected_components(adjacency):
    seen = set()
    components = []
    for node in adjacency:
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    components.sort(key=len, reverse=True)
    return components


def bfs_distances(adjacency, source, allowed_nodes=None):
    allowed = None if allowed_nodes is None else set(allowed_nodes)
    distances = {source: 0}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if allowed is not None and neighbor not in allowed:
                continue
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    return distances


def pairwise_distance_matrix(adjacency, nodes, allowed_nodes):
    matrix = np.full((len(nodes), len(nodes)), np.inf)
    allowed = set(allowed_nodes)
    for row, node in enumerate(nodes):
        distances = bfs_distances(adjacency, node, allowed)
        for col, other in enumerate(nodes):
            value = distances.get(other)
            if value is not None:
                matrix[row, col] = value
    return matrix


def delta_from_quartet(matrix, a, b, c, d):
    sums = sorted(
        [
            matrix[a, b] + matrix[c, d],
            matrix[a, c] + matrix[b, d],
            matrix[a, d] + matrix[b, c],
        ]
    )
    return 0.5 * (sums[2] - sums[1])


def exact_hyperbolicity(matrix):
    max_delta = 0.0
    argmax_quartet = None
    for quartet in itertools.combinations(range(matrix.shape[0]), 4):
        delta = delta_from_quartet(matrix, *quartet)
        if delta > max_delta:
            max_delta = float(delta)
            argmax_quartet = quartet
    return max_delta, argmax_quartet


def approximate_hyperbolicity(matrix, quartet_samples, rng):
    node_count = matrix.shape[0]
    all_indices = list(range(node_count))
    max_delta = 0.0
    argmax_quartet = None
    for _ in range(quartet_samples):
        quartet = tuple(rng.sample(all_indices, 4))
        delta = delta_from_quartet(matrix, *quartet)
        if delta > max_delta:
            max_delta = float(delta)
            argmax_quartet = quartet
    return max_delta, argmax_quartet


def summarize_distances(matrix):
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    finite = upper[np.isfinite(upper)]
    if finite.size == 0:
        return {"sample_diameter": 0, "sample_mean_distance": 0.0}
    return {
        "sample_diameter": int(np.max(finite)),
        "sample_mean_distance": float(np.mean(finite)),
    }


def write_degree_csv(output_dir, degrees):
    path = output_dir / "node_degrees.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node", "degree"])
        for node, degree in sorted(degrees.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([node, degree])
    return path


def write_histogram_csv(output_dir, degree_counts):
    path = output_dir / "degree_histogram.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["degree", "frequency"])
        for degree in sorted(degree_counts):
            writer.writerow([degree, degree_counts[degree]])
    return path


def compute_ccdf(degree_counts):
    degrees = sorted(degree_counts)
    total = sum(degree_counts.values())
    remaining = total
    ccdf_x = []
    ccdf_y = []
    for degree in degrees:
        ccdf_x.append(degree)
        ccdf_y.append(remaining / total)
        remaining -= degree_counts[degree]
    return np.array(ccdf_x), np.array(ccdf_y)


def fit_power_law_tail(degree_counts, xmin=2):
    xs = []
    ys = []
    for degree, count in sorted(degree_counts.items()):
        if degree >= xmin and count > 0:
            xs.append(math.log(degree))
            ys.append(math.log(count))
    if len(xs) < 2:
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    return {
        "xmin": xmin,
        "loglog_frequency_slope": float(slope),
        "approx_power_law_exponent": float(-slope),
        "intercept": float(intercept),
    }


def plot_degree_distribution(output_dir, degree_counts):
    degrees = np.array(sorted(degree_counts))
    counts = np.array([degree_counts[degree] for degree in degrees])
    ccdf_x, ccdf_y = compute_ccdf(degree_counts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(degrees, counts, s=18, alpha=0.75)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Node count")
    axes[0].set_title("Degree Frequency (log-log)")
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].plot(ccdf_x, ccdf_y, marker="o", markersize=3, linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Degree")
    axes[1].set_ylabel("P(K >= k)")
    axes[1].set_title("Degree CCDF (log-log)")
    axes[1].grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    plot_path = output_dir / "degree_distribution.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def analyze_hyperbolicity(adjacency, largest_component, args):
    component_size = len(largest_component)
    rng = random.Random(args.seed)

    if component_size < 4:
        return {
            "mode": "not_applicable",
            "largest_component_size": component_size,
            "delta": 0.0,
            "normalized_delta_by_sample_diameter": 0.0,
            "quartet_samples_evaluated": 0,
            "sampled_nodes": component_size,
        }

    if component_size <= args.exact_component_limit:
        sample_nodes = list(largest_component)
        matrix = pairwise_distance_matrix(adjacency, sample_nodes, largest_component)
        delta, witness = exact_hyperbolicity(matrix)
        distance_summary = summarize_distances(matrix)
        diameter = distance_summary["sample_diameter"]
        return {
            "mode": "exact",
            "largest_component_size": component_size,
            "sampled_nodes": len(sample_nodes),
            "quartet_samples_evaluated": math.comb(len(sample_nodes), 4),
            "delta": delta,
            "normalized_delta_by_sample_diameter": 0.0 if diameter == 0 else delta / diameter,
            "witness_nodes": [sample_nodes[index] for index in witness] if witness else [],
            **distance_summary,
        }

    sample_size = min(args.delta_sample_nodes, component_size)
    sample_nodes = rng.sample(list(largest_component), sample_size)
    matrix = pairwise_distance_matrix(adjacency, sample_nodes, largest_component)
    quartet_budget = min(args.delta_sample_quartets, math.comb(sample_size, 4))
    delta, witness = approximate_hyperbolicity(matrix, quartet_budget, rng)
    distance_summary = summarize_distances(matrix)
    diameter = distance_summary["sample_diameter"]
    return {
        "mode": "sampled_approximation",
        "largest_component_size": component_size,
        "sampled_nodes": sample_size,
        "quartet_samples_evaluated": quartet_budget,
        "delta": delta,
        "normalized_delta_by_sample_diameter": 0.0 if diameter == 0 else delta / diameter,
        "witness_nodes": [sample_nodes[index] for index in witness] if witness else [],
        **distance_summary,
    }


def build_summary(adjacency, directed_triples, relation_counts, self_loops, components, hyperbolicity):
    degrees = {node: len(neighbors) for node, neighbors in adjacency.items()}
    degree_values = np.array(list(degrees.values()), dtype=float)
    degree_counts = Counter(degrees.values())
    largest_component = components[0] if components else []
    largest_component_fraction = 0.0 if not adjacency else len(largest_component) / len(adjacency)
    top_hubs = sorted(degrees.items(), key=lambda item: (-item[1], item[0]))[:20]
    singleton_fraction = float(np.mean(degree_values == 1)) if degree_values.size else 0.0
    low_degree_fraction = float(np.mean(degree_values <= 2)) if degree_values.size else 0.0

    summary = {
        "graph": {
            "nodes": len(adjacency),
            "directed_triples": directed_triples,
            "undirected_edges": int(sum(degrees.values()) // 2),
            "self_loops_ignored": self_loops,
            "connected_components": len(components),
            "largest_component_size": len(largest_component),
            "largest_component_fraction": largest_component_fraction,
            "top_relations": relation_counts.most_common(20),
        },
        "degree": {
            "min": int(np.min(degree_values)) if degree_values.size else 0,
            "max": int(np.max(degree_values)) if degree_values.size else 0,
            "mean": float(np.mean(degree_values)) if degree_values.size else 0.0,
            "median": float(np.median(degree_values)) if degree_values.size else 0.0,
            "p90": float(np.percentile(degree_values, 90)) if degree_values.size else 0.0,
            "p99": float(np.percentile(degree_values, 99)) if degree_values.size else 0.0,
            "singleton_fraction": singleton_fraction,
            "degree_leq_2_fraction": low_degree_fraction,
            "top_hubs": top_hubs,
        },
        "hyperbolicity": hyperbolicity,
        "power_law_hint": fit_power_law_tail(degree_counts),
    }
    return summary, degrees, degree_counts


def print_summary(summary, output_dir):
    graph = summary["graph"]
    degree = summary["degree"]
    hyper = summary["hyperbolicity"]

    print(f"Analysis written to: {output_dir}")
    print(
        "Graph: "
        f"{graph['nodes']} nodes, {graph['undirected_edges']} undirected edges, "
        f"{graph['connected_components']} components"
    )
    print(
        "Degree: "
        f"mean={degree['mean']:.2f}, median={degree['median']:.2f}, "
        f"p90={degree['p90']:.2f}, p99={degree['p99']:.2f}, max={degree['max']}"
    )
    print(
        "Tail: "
        f"{degree['singleton_fraction']:.2%} degree-1 nodes, "
        f"{degree['degree_leq_2_fraction']:.2%} nodes with degree <= 2"
    )
    print(
        "Hyperbolicity: "
        f"mode={hyper['mode']}, delta={hyper['delta']:.3f}, "
        f"delta/diameter={hyper['normalized_delta_by_sample_diameter']:.3f}, "
        f"sample_diameter={hyper.get('sample_diameter', 0)}"
    )


def main():
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adjacency, directed_triples, relation_counts, self_loops = load_graph(dataset_dir, args.splits)
    components = connected_components(adjacency)
    largest_component = components[0] if components else []
    hyperbolicity = analyze_hyperbolicity(adjacency, largest_component, args)
    summary, degrees, degree_counts = build_summary(
        adjacency, directed_triples, relation_counts, self_loops, components, hyperbolicity
    )

    plot_path = plot_degree_distribution(output_dir, degree_counts)
    degrees_path = write_degree_csv(output_dir, degrees)
    histogram_path = write_histogram_csv(output_dir, degree_counts)
    summary["artifacts"] = {
        "degree_distribution_plot": str(plot_path),
        "node_degrees_csv": str(degrees_path),
        "degree_histogram_csv": str(histogram_path),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print_summary(summary, output_dir)


if __name__ == "__main__":
    main()
