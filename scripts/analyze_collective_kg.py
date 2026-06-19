#!/usr/bin/env python3
"""Deep analysis and publication-ready figures for merged KG splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

Triple = Tuple[str, str, str]

URI_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("https://www.imgt.org/imgt-ontology#", "imgt:"),
    ("http://www.imgt.org/imgt-ontology#", "imgt:"),
    ("http://purl.obolibrary.org/obo/", "obo:"),
    ("https://purl.obolibrary.org/obo/", "obo:"),
    ("http://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/", "HGNC:"),
    ("https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/", "HGNC:"),
    ("http://www.orpha.net/ORDO/", "ORDO:"),
    ("https://www.orpha.net/ORDO/", "ORDO:"),
    ("http://identifiers.org/drugbank/", "drugbank:"),
    ("https://identifiers.org/drugbank/", "drugbank:"),
    ("http://identifiers.org/uniprot/", "uniprot:"),
    ("https://identifiers.org/uniprot/", "uniprot:"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze train/valid/test KGs as one merged graph and render figures."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/collective_kg_analysis"),
    )
    parser.add_argument("--top-relations", type=int, default=15)
    parser.add_argument("--top-families", type=int, default=12)
    parser.add_argument("--network-nodes", type=int, default=120)
    parser.add_argument(
        "--mab-node",
        type=str,
        default=None,
        help="Specific mAb node to center a local topology snapshot on (e.g., imgt:mAb_123).",
    )
    parser.add_argument(
        "--mab-radius",
        type=int,
        default=2,
        help="Hop radius for the mAb ego network.",
    )
    parser.add_argument(
        "--mab-nodes",
        type=int,
        default=90,
        help="Maximum number of nodes to retain in the mAb local snapshot.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compact_term(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    if not value.startswith("http://") and not value.startswith("https://"):
        return value
    for prefix, replacement in URI_PREFIXES:
        if value.startswith(prefix):
            return replacement + value[len(prefix) :]
    if "#" in value:
        return value.rsplit("#", 1)[1]
    return value.rstrip("/").rsplit("/", 1)[-1]


def local_name(value: str) -> str:
    compact = compact_term(value)
    if ":" in compact:
        return compact.split(":", 1)[1]
    return compact


def split_family(entity: str) -> str:
    token = local_name(entity)
    if token.isdigit():
        return "NumericId"
    if token.startswith("mAb_"):
        return "mAb"
    if token.startswith("StudyProduct_"):
        return "StudyProduct"
    if token.startswith("Product_"):
        return "Product"
    if token.startswith("Decision_"):
        return "Decision"
    if token.startswith("Construct_"):
        return "Construct"
    if token.startswith("Segment_"):
        return "Segment"
    if token.startswith("Clone_"):
        return "Clone"
    if token.startswith("MOA_"):
        return "MOA"
    if token.startswith("Phase_"):
        return "Phase"
    if token.startswith("HGNC:"):
        return "HGNC"
    if entity.startswith("HGNC:"):
        return "HGNC"
    if entity.startswith(("obo:MONDO_", "doid:", "ORDO:")):
        return "Disease"
    if token.startswith(
        (
            "Cancers_",
            "Cancer_",
            "Carcinoma_",
            "Lymphoma_",
            "Leukemia_",
            "Melanoma_",
            "Myeloma_",
            "Solid_tumors",
        )
    ):
        return "Disease"
    if token in {"FDA", "EMA"} or token.endswith(
        ("_Inc", "_Ltd", "_GmbH", "_Corp", "_Corporation", "_PLC", "_SA", "_SAS", "_LLC")
    ):
        return "Organisation"
    if "_" in token:
        return token.split("_", 1)[0]
    return token


def short_label(value: str, max_len: int = 42) -> str:
    x = local_name(value).replace("_", " ")
    if len(x) <= max_len:
        return x
    return x[: max_len - 1] + "..."


def load_split(path: Path) -> List[Triple]:
    triples: List[Triple] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_number} does not have 3 tab-separated fields")
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def gini(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    v = np.sort(values.astype(float))
    if np.all(v == 0.0):
        return 0.0
    n = v.size
    i = np.arange(1, n + 1, dtype=float)
    return float((np.sum((2 * i - n - 1) * v)) / (n * np.sum(v)))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "#f6f7fb",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#f6f7fb",
            "axes.edgecolor": "#d8dce8",
            "axes.labelcolor": "#1f2430",
            "xtick.color": "#2b3342",
            "ytick.color": "#2b3342",
            "grid.color": "#e6e9f2",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def build_graph_objects(triples: Sequence[Triple]) -> Tuple[nx.DiGraph, nx.Graph]:
    dg = nx.DiGraph()
    ug = nx.Graph()
    for h, r, t in triples:
        if dg.has_edge(h, t):
            dg[h][t]["weight"] += 1
            dg[h][t]["relations"][r] += 1
        else:
            dg.add_edge(h, t, weight=1, relations=Counter({r: 1}))
        if ug.has_edge(h, t):
            ug[h][t]["weight"] += 1
        else:
            ug.add_edge(h, t, weight=1)
    return dg, ug


def analyze_merged_graph(triples: Sequence[Triple]) -> Dict[str, object]:
    relation_counts: Counter[str] = Counter()
    entity_mentions: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    entities: Set[str] = set()

    rel_heads: DefaultDict[str, Set[str]] = defaultdict(set)
    rel_tails: DefaultDict[str, Set[str]] = defaultdict(set)
    rel_pairs: DefaultDict[str, Set[Tuple[str, str]]] = defaultdict(set)

    family_matrix: DefaultDict[str, Counter[str]] = defaultdict(Counter)
    relation_family_matrix: DefaultDict[str, Counter[Tuple[str, str]]] = defaultdict(Counter)

    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relation_counts[r] += 1
        entity_mentions[h] += 1
        entity_mentions[t] += 1
        rel_heads[r].add(h)
        rel_tails[r].add(t)
        rel_pairs[r].add((h, t))

        fh = split_family(h)
        ft = split_family(t)
        family_counts[fh] += 1
        family_counts[ft] += 1
        family_matrix[fh][ft] += 1
        relation_family_matrix[r][(fh, ft)] += 1

    dg, ug = build_graph_objects(triples)
    in_degree = np.array([d for _, d in dg.in_degree()], dtype=float)
    out_degree = np.array([d for _, d in dg.out_degree()], dtype=float)
    total_degree = in_degree + out_degree

    components = sorted(nx.connected_components(ug), key=len, reverse=True)
    component_sizes = [len(c) for c in components]

    relation_stats: List[Dict[str, object]] = []
    for rel, count in relation_counts.most_common():
        n_heads = len(rel_heads[rel])
        n_tails = len(rel_tails[rel])
        n_pairs = len(rel_pairs[rel])
        tph = count / n_heads if n_heads else 0.0
        hpt = count / n_tails if n_tails else 0.0
        pair_density = n_pairs / (n_heads * n_tails) if n_heads and n_tails else 0.0
        dominant_family_pair = ("Unknown", "Unknown")
        dominant_pair_count = 0
        if relation_family_matrix[rel]:
            dominant_family_pair, dominant_pair_count = relation_family_matrix[rel].most_common(1)[0]
        relation_stats.append(
            {
                "relation": rel,
                "count": count,
                "num_heads": n_heads,
                "num_tails": n_tails,
                "num_unique_pairs": n_pairs,
                "tails_per_head": tph,
                "heads_per_tail": hpt,
                "pair_density": pair_density,
                "dominant_head_family": dominant_family_pair[0],
                "dominant_tail_family": dominant_family_pair[1],
                "dominant_family_pair_count": dominant_pair_count,
            }
        )

    top_entities = sorted(dg.degree(), key=lambda x: x[1], reverse=True)[:200]

    return {
        "entities": entities,
        "relation_counts": relation_counts,
        "entity_mentions": entity_mentions,
        "family_counts": family_counts,
        "family_matrix": family_matrix,
        "directed_graph": dg,
        "undirected_graph": ug,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": total_degree,
        "components": components,
        "component_sizes": component_sizes,
        "relation_stats": relation_stats,
        "top_entities": top_entities,
        "summary": {
            "triples": len(triples),
            "entities": len(entities),
            "relations": len(relation_counts),
            "avg_in_degree": float(np.mean(in_degree)) if in_degree.size else 0.0,
            "avg_out_degree": float(np.mean(out_degree)) if out_degree.size else 0.0,
            "avg_total_degree": float(np.mean(total_degree)) if total_degree.size else 0.0,
            "median_total_degree": float(np.median(total_degree)) if total_degree.size else 0.0,
            "max_total_degree": int(np.max(total_degree)) if total_degree.size else 0,
            "degree_gini": gini(total_degree),
            "num_components": len(components),
            "largest_component_nodes": component_sizes[0] if component_sizes else 0,
            "largest_component_fraction": float(component_sizes[0] / len(entities))
            if component_sizes and entities
            else 0.0,
            "self_loops": int(nx.number_of_selfloops(ug)),
        },
    }


def save_tables(analysis: Dict[str, object], output_dir: Path) -> None:
    relation_stats = analysis["relation_stats"]
    family_counts: Counter[str] = analysis["family_counts"]
    component_sizes: List[int] = analysis["component_sizes"]
    top_entities: List[Tuple[str, int]] = analysis["top_entities"]
    relation_counts: Counter[str] = analysis["relation_counts"]

    with (output_dir / "relation_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(relation_stats[0].keys()) if relation_stats else [])
        if relation_stats:
            writer.writeheader()
            writer.writerows(relation_stats)

    with (output_dir / "family_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "mentions"])
        for fam, count in family_counts.most_common():
            writer.writerow([fam, count])

    with (output_dir / "component_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["component_rank", "num_nodes"])
        for i, size in enumerate(component_sizes, start=1):
            writer.writerow([i, size])

    with (output_dir / "top_entities_by_degree.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entity", "degree"])
        for ent, degree in top_entities:
            writer.writerow([ent, degree])

    with (output_dir / "top_relations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relation", "count"])
        for rel, count in relation_counts.most_common(100):
            writer.writerow([rel, count])


def save_summary_text(analysis: Dict[str, object], output_dir: Path) -> None:
    s = analysis["summary"]
    relation_counts: Counter[str] = analysis["relation_counts"]
    family_counts: Counter[str] = analysis["family_counts"]
    top_entities: List[Tuple[str, int]] = analysis["top_entities"]
    rel_stats_sorted = sorted(analysis["relation_stats"], key=lambda x: x["count"], reverse=True)

    lines: List[str] = []
    lines.append("Collective KG analysis (train+valid+test merged)")
    lines.append(f"Triples: {s['triples']}")
    lines.append(f"Entities: {s['entities']}")
    lines.append(f"Relations: {s['relations']}")
    lines.append(f"Components: {s['num_components']}")
    lines.append(
        f"Largest component: {s['largest_component_nodes']} nodes ({100.0 * s['largest_component_fraction']:.2f}%)"
    )
    lines.append(f"Average degree: {s['avg_total_degree']:.3f}")
    lines.append(f"Median degree: {s['median_total_degree']:.3f}")
    lines.append(f"Max degree: {s['max_total_degree']}")
    lines.append(f"Degree Gini: {s['degree_gini']:.3f}")
    lines.append("")
    lines.append("Top relations")
    for rel, count in relation_counts.most_common(20):
        lines.append(f"- {rel}: {count}")
    lines.append("")
    lines.append("Top families (by mention frequency)")
    for fam, count in family_counts.most_common(20):
        lines.append(f"- {fam}: {count}")
    lines.append("")
    lines.append("Highest-degree entities")
    for ent, degree in top_entities[:20]:
        lines.append(f"- {ent}: {degree}")
    lines.append("")
    lines.append("Relation arity highlights (top 15 by count)")
    for row in rel_stats_sorted[:15]:
        lines.append(
            "- {relation}: count={count}, heads={num_heads}, tails={num_tails}, "
            "tails/head={tails_per_head:.3f}, heads/tail={heads_per_tail:.3f}, pair_density={pair_density:.4f}".format(
                **row
            )
        )

    with (output_dir / "analysis_report.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", pad=10, color="#162032")


def make_overview_figure(
    analysis: Dict[str, object],
    output_dir: Path,
    top_relations: int,
    top_families: int,
) -> None:
    relation_counts: Counter[str] = analysis["relation_counts"]
    family_counts: Counter[str] = analysis["family_counts"]
    total_degree: np.ndarray = analysis["total_degree"]
    in_degree: np.ndarray = analysis["in_degree"]
    out_degree: np.ndarray = analysis["out_degree"]
    rel_stats_sorted = sorted(analysis["relation_stats"], key=lambda x: x["count"], reverse=True)
    s = analysis["summary"]

    top_rel = relation_counts.most_common(top_relations)
    top_fam = family_counts.most_common(top_families)

    top_family_names = [fam for fam, _ in top_fam]
    fm = np.zeros((len(top_family_names), len(top_family_names)), dtype=float)
    family_matrix: DefaultDict[str, Counter[str]] = analysis["family_matrix"]
    for i, fh in enumerate(top_family_names):
        for j, ft in enumerate(top_family_names):
            fm[i, j] = family_matrix[fh][ft]

    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, hspace=0.15, wspace=0.12)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

    _panel_title(ax0, "A. Global Summary")
    ax0.axis("off")
    ax0.text(
        0.02,
        0.95,
        "Collective IMGT KG\n(train + valid + test merged)",
        fontsize=16,
        fontweight="bold",
        va="top",
        color="#0f1728",
    )
    key_lines = [
        f"Triples: {s['triples']:,}",
        f"Entities: {s['entities']:,}",
        f"Relations: {s['relations']:,}",
        f"Connected components: {s['num_components']:,}",
        f"Largest component: {s['largest_component_nodes']:,} nodes ({100.0 * s['largest_component_fraction']:.1f}%)",
        f"Average in-degree: {s['avg_in_degree']:.2f}",
        f"Average out-degree: {s['avg_out_degree']:.2f}",
        f"Median total degree: {s['median_total_degree']:.1f}",
        f"Max total degree: {s['max_total_degree']:,}",
        f"Degree inequality (Gini): {s['degree_gini']:.3f}",
        f"Self-loops: {s['self_loops']:,}",
    ]
    ax0.text(0.02, 0.78, "\n".join(key_lines), fontsize=11.5, va="top", linespacing=1.5, color="#20263a")

    _panel_title(ax1, "B. Most Frequent Relations")
    rel_names = [short_label(r, 36) for r, _ in reversed(top_rel)]
    rel_vals = [c for _, c in reversed(top_rel)]
    cmap = mpl.colormaps["viridis"]
    colors = [cmap(0.25 + 0.7 * i / max(1, len(rel_vals) - 1)) for i in range(len(rel_vals))]
    ax1.barh(rel_names, rel_vals, color=colors, edgecolor="#20263a", linewidth=0.2)
    ax1.set_xlabel("Triple count")
    ax1.grid(axis="x", linestyle="-", alpha=0.65)

    _panel_title(ax2, "C. Entity Family Frequency")
    fam_names = [short_label(f, 28) for f, _ in reversed(top_fam)]
    fam_vals = [c for _, c in reversed(top_fam)]
    plasma = mpl.colormaps["plasma"]
    colors2 = [plasma(0.2 + 0.7 * i / max(1, len(fam_vals) - 1)) for i in range(len(fam_vals))]
    ax2.barh(fam_names, fam_vals, color=colors2, edgecolor="#20263a", linewidth=0.2)
    ax2.set_xlabel("Entity mentions in triples")
    ax2.grid(axis="x", linestyle="-", alpha=0.65)

    _panel_title(ax3, "D. Degree Distribution (CCDF)")
    degrees = np.sort(total_degree[total_degree > 0])
    if degrees.size > 0:
        y = 1.0 - np.arange(1, degrees.size + 1) / degrees.size
        y = np.clip(y, 1e-6, 1.0)
        ax3.plot(degrees, y, color="#124e96", linewidth=2.5)
        ax3.scatter(degrees[:: max(1, degrees.size // 500)], y[:: max(1, degrees.size // 500)], s=8, color="#2f7cd4")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Total degree")
    ax3.set_ylabel("CCDF: P(Degree ≥ k)")
    ax3.grid(True, which="both", linestyle="-", alpha=0.5)

    _panel_title(ax4, "E. Relation Arity Footprint")
    x = np.array([max(1e-6, row["tails_per_head"]) for row in rel_stats_sorted], dtype=float)
    y = np.array([max(1e-6, row["heads_per_tail"]) for row in rel_stats_sorted], dtype=float)
    sizes = np.array([row["count"] for row in rel_stats_sorted], dtype=float)
    sizes = 30 + 720 * (sizes / sizes.max())
    cvals = np.log10(np.array([row["count"] for row in rel_stats_sorted], dtype=float) + 1.0)
    sc = ax4.scatter(
        x,
        y,
        s=sizes,
        c=cvals,
        cmap="cividis",
        alpha=0.8,
        edgecolors="#1e2331",
        linewidths=0.3,
    )
    for row in rel_stats_sorted[:8]:
        ax4.annotate(
            short_label(row["relation"], 24),
            (max(1e-6, row["tails_per_head"]), max(1e-6, row["heads_per_tail"])),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            color="#192239",
        )
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("Tails per head (avg)")
    ax4.set_ylabel("Heads per tail (avg)")
    ax4.grid(True, which="both", linestyle="-", alpha=0.5)
    cbar = plt.colorbar(sc, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("log10(relation count + 1)")

    _panel_title(ax5, "F. Family-to-Family Interaction Matrix")
    im = ax5.imshow(np.log10(fm + 1.0), cmap="magma", aspect="auto")
    ax5.set_xticks(np.arange(len(top_family_names)))
    ax5.set_yticks(np.arange(len(top_family_names)))
    ax5.set_xticklabels([short_label(f, 16) for f in top_family_names], rotation=55, ha="right", fontsize=8)
    ax5.set_yticklabels([short_label(f, 16) for f in top_family_names], fontsize=8)
    cbar2 = plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    cbar2.set_label("log10(count + 1)")

    fig.suptitle(
        "Collective Knowledge Graph Structural Analysis",
        fontsize=18,
        fontweight="bold",
        color="#0f1728",
        y=1.02,
    )

    fig_path_png = output_dir / "collective_kg_overview.png"
    fig_path_pdf = output_dir / "collective_kg_overview.pdf"
    fig_path_svg = output_dir / "collective_kg_overview.svg"
    fig.savefig(fig_path_png, dpi=320, bbox_inches="tight")
    fig.savefig(fig_path_pdf, dpi=320, bbox_inches="tight")
    fig.savefig(fig_path_svg, dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_network_figure(
    analysis: Dict[str, object],
    output_dir: Path,
    node_cap: int,
    seed: int,
) -> None:
    ug: nx.Graph = analysis["undirected_graph"]
    dg: nx.DiGraph = analysis["directed_graph"]
    components: List[Set[str]] = analysis["components"]

    if not components:
        return

    largest_nodes = list(components[0])
    sub = ug.subgraph(largest_nodes)
    deg_sorted = sorted(sub.degree(), key=lambda x: x[1], reverse=True)
    if not deg_sorted:
        return

    # Build a connected, hub-centered view using BFS expansion.
    root = deg_sorted[0][0]
    bfs_nodes: List[str] = []
    for node in nx.bfs_tree(sub, source=root):
        bfs_nodes.append(node)
        if len(bfs_nodes) >= node_cap:
            break
    selected: Set[str] = set(bfs_nodes)
    if len(selected) < node_cap:
        for n, _ in deg_sorted:
            selected.add(n)
            if len(selected) >= node_cap:
                break

    g = sub.subgraph(selected).copy()
    # Keep only the largest connected chunk in case the cap cut the fringe awkwardly.
    if g.number_of_nodes() > 0 and not nx.is_connected(g):
        largest_cc_nodes = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc_nodes).copy()
    if g.number_of_nodes() == 0:
        return

    if g.number_of_nodes() <= 180:
        pos = nx.kamada_kawai_layout(g)
    else:
        pos = nx.spring_layout(
            g,
            seed=seed,
            k=0.85 / math.sqrt(max(1, g.number_of_nodes())),
            iterations=300,
        )

    fams = sorted({split_family(n) for n in g.nodes()})
    palette = mpl.colormaps["tab20"]
    family_color = {fam: palette(i / max(1, len(fams) - 1)) for i, fam in enumerate(fams)}
    node_colors = [family_color[split_family(n)] for n in g.nodes()]
    node_sizes = [38 + 22 * math.sqrt(max(1, dg.degree(n))) for n in g.nodes()]

    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
    ax.set_facecolor("#ffffff")
    ax.set_title(
        "Local Topology Snapshot in Largest Connected Component",
        loc="left",
        fontweight="bold",
        fontsize=15,
        color="#111829",
        pad=10,
    )

    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        width=1.0,
        alpha=0.24,
        edge_color="#5c6a84",
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.4,
        edgecolors="#121826",
    )

    high_nodes = sorted(g.degree(), key=lambda x: x[1], reverse=True)[:14]
    labels = {n: short_label(n, 20) for n, _ in high_nodes}
    text_items = nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, font_color="#101626", ax=ax)
    for text in text_items.values():
        text.set_bbox(
            dict(
                facecolor="#ffffff",
                edgecolor="#d7dceb",
                alpha=0.82,
                boxstyle="round,pad=0.14",
                linewidth=0.5,
            )
        )

    legend_handles = []
    for fam in fams[:14]:
        legend_handles.append(
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=family_color[fam],
                markeredgecolor="#121826",
                markersize=8,
                label=short_label(fam, 22),
            )
        )
    ax.legend(
        handles=legend_handles,
        title="Entity family",
        loc="upper right",
        frameon=True,
        facecolor="#f6f7fb",
        edgecolor="#d8dce8",
        fontsize=9,
        title_fontsize=10,
    )

    # Tighten bounds so one outlier does not collapse readability.
    xs = np.array([xy[0] for xy in pos.values()], dtype=float)
    ys = np.array([xy[1] for xy in pos.values()], dtype=float)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    dx = max(1e-6, x_max - x_min)
    dy = max(1e-6, y_max - y_min)
    ax.set_xlim(x_min - 0.08 * dx, x_max + 0.08 * dx)
    ax.set_ylim(y_min - 0.08 * dy, y_max + 0.08 * dy)
    ax.axis("off")

    fig_path_png = output_dir / "collective_kg_local_topology.png"
    fig_path_pdf = output_dir / "collective_kg_local_topology.pdf"
    fig_path_svg = output_dir / "collective_kg_local_topology.svg"
    fig.savefig(fig_path_png, dpi=320, bbox_inches="tight")
    fig.savefig(fig_path_pdf, dpi=320, bbox_inches="tight")
    fig.savefig(fig_path_svg, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _sanitize_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def choose_mab_node(dg: nx.DiGraph, requested: str | None) -> str:
    mabs = [n for n in dg.nodes() if split_family(n) == "mAb"]
    if not mabs:
        raise ValueError("No mAb nodes found in graph.")
    if requested is not None:
        if requested in dg.nodes() and split_family(requested) == "mAb":
            return requested
        raise ValueError(f"Requested --mab-node '{requested}' is missing or not typed as mAb.")
    return max(mabs, key=lambda n: dg.degree(n))


def make_mab_local_snapshot(
    analysis: Dict[str, object],
    output_dir: Path,
    requested_mab: str | None,
    radius: int,
    node_cap: int,
    seed: int,
) -> str:
    ug: nx.Graph = analysis["undirected_graph"]
    dg: nx.DiGraph = analysis["directed_graph"]
    center = choose_mab_node(dg, requested_mab)

    ego_nodes = set(nx.ego_graph(ug, center, radius=max(1, radius)).nodes())
    if len(ego_nodes) > node_cap:
        ranked = sorted(ego_nodes, key=lambda n: dg.degree(n), reverse=True)
        keep = set(ranked[: max(10, node_cap - 1)])
        keep.add(center)
        ego_nodes = keep

    g = ug.subgraph(ego_nodes).copy()
    if g.number_of_nodes() == 0:
        raise ValueError("Empty ego graph for selected mAb.")

    if g.number_of_nodes() <= 120:
        pos = nx.kamada_kawai_layout(g)
    else:
        pos = nx.spring_layout(g, seed=seed, k=0.8 / math.sqrt(g.number_of_nodes()), iterations=250)

    fams = sorted({split_family(n) for n in g.nodes()})
    palette = mpl.colormaps["Set2"]
    family_color = {fam: palette(i / max(1, len(fams) - 1)) for i, fam in enumerate(fams)}
    node_colors = [family_color[split_family(n)] for n in g.nodes()]
    node_sizes = [40 + 22 * math.sqrt(max(1, dg.degree(n))) for n in g.nodes()]
    node_sizes = [s * 1.9 if n == center else s for n, s in zip(g.nodes(), node_sizes)]

    fig, ax = plt.subplots(figsize=(13, 10), constrained_layout=True)
    ax.set_facecolor("#ffffff")
    ax.set_title(
        f"Local Topology Snapshot around {short_label(center, 28)}",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color="#111829",
        pad=10,
    )

    nx.draw_networkx_edges(g, pos, ax=ax, width=1.0, alpha=0.30, edge_color="#5f6f87")
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.6,
        edgecolors="#121826",
    )

    label_nodes = sorted(g.degree(), key=lambda x: x[1], reverse=True)[:14]
    if center not in {n for n, _ in label_nodes}:
        label_nodes = [(center, g.degree(center))] + label_nodes[:-1]
    labels = {n: short_label(n, 22) for n, _ in label_nodes}
    text_items = nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, font_color="#101626", ax=ax)
    for text in text_items.values():
        text.set_bbox(
            dict(facecolor="#ffffff", edgecolor="#d7dceb", alpha=0.84, boxstyle="round,pad=0.14", linewidth=0.5)
        )

    center_neighbors = list(g.neighbors(center))
    edge_labels = {}
    for nbr in center_neighbors[:8]:
        rel_counter = Counter()
        if dg.has_edge(center, nbr):
            rel_counter.update(dg[center][nbr].get("relations", {}))
        if dg.has_edge(nbr, center):
            rel_counter.update(dg[nbr][center].get("relations", {}))
        if rel_counter:
            rel = rel_counter.most_common(1)[0][0]
            edge_labels[(center, nbr)] = short_label(rel, 18)
    if edge_labels:
        nx.draw_networkx_edge_labels(
            g,
            pos,
            edge_labels=edge_labels,
            font_size=7,
            font_color="#24314a",
            rotate=False,
            ax=ax,
            bbox=dict(facecolor="#f5f7fb", edgecolor="#dce2ef", alpha=0.82, pad=0.2),
        )

    legend_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=family_color[fam],
            markeredgecolor="#121826",
            markersize=8,
            label=short_label(fam, 20),
        )
        for fam in fams
    ]
    ax.legend(
        handles=legend_handles,
        title="Entity family",
        loc="upper right",
        frameon=True,
        facecolor="#f6f7fb",
        edgecolor="#d8dce8",
        fontsize=9,
        title_fontsize=10,
    )

    ax.text(
        0.015,
        0.03,
        f"center={center} | radius={max(1, radius)} | nodes={g.number_of_nodes()} | edges={g.number_of_edges()}",
        transform=ax.transAxes,
        fontsize=9,
        color="#2a3347",
    )
    ax.axis("off")

    slug = _sanitize_name(local_name(center))
    fig_png = output_dir / f"mab_local_topology_{slug}.png"
    fig_pdf = output_dir / f"mab_local_topology_{slug}.pdf"
    fig_svg = output_dir / f"mab_local_topology_{slug}.svg"
    fig.savefig(fig_png, dpi=320, bbox_inches="tight")
    fig.savefig(fig_pdf, dpi=320, bbox_inches="tight")
    fig.savefig(fig_svg, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return center


def main() -> None:
    args = parse_args()
    split_paths = [args.data_dir / "train.txt", args.data_dir / "valid.txt", args.data_dir / "test.txt"]
    missing = [str(p) for p in split_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing split file(s): {missing}")

    triples: List[Triple] = []
    split_summary: Dict[str, int] = {}
    for p in split_paths:
        data = load_split(p)
        split_summary[p.stem] = len(data)
        triples.extend(data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    analysis = analyze_merged_graph(triples)
    analysis["split_summary"] = split_summary

    metrics = {
        "split_summary": split_summary,
        "summary": analysis["summary"],
        "top_relations": [
            {"relation": rel, "count": count}
            for rel, count in analysis["relation_counts"].most_common(100)
        ],
        "top_families": [
            {"family": fam, "mentions": count}
            for fam, count in analysis["family_counts"].most_common(100)
        ],
    }
    with (args.output_dir / "collective_kg_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=True)

    save_tables(analysis, args.output_dir)
    save_summary_text(analysis, args.output_dir)
    make_overview_figure(
        analysis=analysis,
        output_dir=args.output_dir,
        top_relations=args.top_relations,
        top_families=args.top_families,
    )
    make_network_figure(
        analysis=analysis,
        output_dir=args.output_dir,
        node_cap=args.network_nodes,
        seed=args.seed,
    )
    selected_mab = make_mab_local_snapshot(
        analysis=analysis,
        output_dir=args.output_dir,
        requested_mab=args.mab_node,
        radius=args.mab_radius,
        node_cap=args.mab_nodes,
        seed=args.seed,
    )

    print(f"Wrote outputs to {args.output_dir}")
    print(f"- selected mAb center: {selected_mab}")
    for name in (
        "collective_kg_overview.png",
        "collective_kg_overview.pdf",
        "collective_kg_overview.svg",
        "collective_kg_local_topology.png",
        "collective_kg_local_topology.pdf",
        "collective_kg_local_topology.svg",
        f"mab_local_topology_{_sanitize_name(local_name(selected_mab))}.png",
        f"mab_local_topology_{_sanitize_name(local_name(selected_mab))}.pdf",
        f"mab_local_topology_{_sanitize_name(local_name(selected_mab))}.svg",
        "collective_kg_metrics.json",
        "analysis_report.txt",
        "relation_stats.csv",
        "family_stats.csv",
        "component_stats.csv",
        "top_entities_by_degree.csv",
        "top_relations.csv",
    ):
        print(f"- {args.output_dir / name}")


if __name__ == "__main__":
    main()
