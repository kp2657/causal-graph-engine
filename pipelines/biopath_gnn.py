"""
biopath_gnn.py — BioPathNet-style Biologically Relevant Graph (BRG) diffusion.

Key insight from BioPathNet (He et al. 2024): adding protein-pathway interaction
edges to message passing substantially improves drug repurposing and synthetic
lethality prediction, even when those edges are not directly part of the
prediction task.

We implement this as Random Walk with Restart (RWR) on the BRG, seeded from
high-γ genes from the Ota pipeline.  The result is a ranked list of novel
gene-disease link candidates not yet covered by Perturb-seq or eQTL evidence.

Properties:
  - Inductive: works for any new disease by reseeding with that disease's
    causal anchor genes — no model retraining required.
  - Topology-preserving: PPI + pathway relay nodes capture functional
    neighborhoods invisible to gene-level analysis alone.
  - Query-conditioned: disease-specific seed weights propagate disease-relevant
    signal through the BRG, suppressing globally hub-connected but
    disease-irrelevant genes.

BRG node types:  Gene | Pathway (PW: prefix) | Drug (DRUG: prefix)
BRG edge types:
  Gene  <-> Gene     PPI (STRING score-weighted)
  Gene  <-> Pathway  Reactome membership (Gene and PW: relay node)
  Drug  ->  Gene     drug-target (ChEMBL / Open Targets)
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import NamedTuple

RWR_ALPHA       = 0.15   # restart probability
RWR_N_ITER      = 50     # max iterations
RWR_CONVERGENCE = 1e-6
BRG_TOP_K       = 20     # top novel candidates returned

EDGE_WEIGHT: dict[str, float] = {
    "ppi_high":       1.0,   # STRING >= 800
    "ppi_medium":     0.7,   # STRING 700-799
    "pathway_member": 0.9,   # Reactome membership
    "drug_target":    0.8,
}


def build_brg(
    ppi_edges: list[dict],
    pathway_gene_map: dict[str, list[str]],
    drug_target_pairs: list[dict],
    known_disease_genes: list[str],
) -> dict[str, list[tuple[str, float]]]:
    """
    Build the Biologically Relevant Graph adjacency list from three edge sources.

    Args:
        ppi_edges:            List of PPI edge dicts with protein_a/b or gene_a/b keys
                              and a combined_score field (STRING scale 0-1000).
        pathway_gene_map:     {pathway_id: [gene_symbol, ...]} from Reactome.
        drug_target_pairs:    List of {drug, gene} dicts from ChEMBL / Open Targets.
        known_disease_genes:  Seed gene list (used to anchor the graph; not filtered here).

    Returns:
        Column-normalised adjacency list: {node: [(neighbour, weight), ...]}.
    """
    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)

    # PPI edges — bidirectional
    for edge in ppi_edges:
        src = edge.get("protein_a") or edge.get("gene_a") or edge.get("protein1") or ""
        dst = edge.get("protein_b") or edge.get("gene_b") or edge.get("protein2") or ""
        if not src or not dst:
            continue
        score = float(edge.get("combined_score") or edge.get("score") or 0)
        if score >= 800:
            w = EDGE_WEIGHT["ppi_high"] * (score / 1000)
        else:
            w = EDGE_WEIGHT["ppi_medium"] * (score / 1000)
        adj[src].append((dst, w))
        adj[dst].append((src, w))

    # Pathway edges — bidirectional through relay node PW:<pathway_id>
    for pathway_id, genes in pathway_gene_map.items():
        relay = f"PW:{pathway_id}"
        w = EDGE_WEIGHT["pathway_member"]
        for gene in genes:
            adj[gene].append((relay, w))
            adj[relay].append((gene, w))

    # Drug-target edges — bidirectional between DRUG:<drug> and gene
    for pair in drug_target_pairs:
        drug = pair.get("drug") or pair.get("drug_name") or ""
        gene = pair.get("gene") or pair.get("target_gene") or pair.get("gene_symbol") or ""
        if not drug or not gene:
            continue
        drug_node = f"DRUG:{drug}"
        w = EDGE_WEIGHT["drug_target"]
        adj[drug_node].append((gene, w))
        adj[gene].append((drug_node, w))

    # Column-normalise: divide each neighbour weight by the total out-weight of the node
    adj_norm: dict[str, list[tuple[str, float]]] = {}
    for node, neighbors in adj.items():
        total = sum(w for _, w in neighbors)
        if total == 0:
            adj_norm[node] = neighbors
        else:
            adj_norm[node] = [(n, w / total) for n, w in neighbors]

    return adj_norm


def run_rwr(
    adj: dict[str, list[tuple[str, float]]],
    seed_scores: dict[str, float],
    alpha: float = RWR_ALPHA,
    n_iter: int = RWR_N_ITER,
    convergence: float = RWR_CONVERGENCE,
) -> dict[str, float]:
    """
    Random Walk with Restart on the BRG.

    s(t+1) = alpha * seed + (1 - alpha) * A * s(t)

    Args:
        adj:          Column-normalised adjacency list from build_brg.
        seed_scores:  {gene: weight} — unnormalised seed distribution.
        alpha:        Restart probability (default 0.15).
        n_iter:       Maximum iterations (default 50).
        convergence:  Max-change stopping threshold (default 1e-6).

    Returns:
        Stationary distribution {node: score}.
    """
    all_nodes = set(adj.keys()) | set(seed_scores.keys())

    seed_total = sum(seed_scores.values())
    if seed_total == 0:
        seed_norm = {n: 1.0 / len(seed_scores) for n in seed_scores} if seed_scores else {}
    else:
        seed_norm = {n: v / seed_total for n, v in seed_scores.items()}

    scores: dict[str, float] = {n: seed_norm.get(n, 0.0) for n in all_nodes}

    for _ in range(n_iter):
        new_scores: dict[str, float] = {n: alpha * seed_norm.get(n, 0.0) for n in all_nodes}

        for src in adj:
            for dst, w in adj[src]:
                new_scores[dst] = new_scores.get(dst, 0.0) + (1 - alpha) * scores.get(src, 0.0) * w

        max_change = max(abs(new_scores.get(n, 0.0) - scores.get(n, 0.0)) for n in all_nodes)
        scores = new_scores

        if max_change < convergence:
            break

    return scores


def score_novel_links(
    rwr_scores: dict[str, float],
    known_genes: list[str],
    top_k: int = BRG_TOP_K,
) -> list[dict]:
    """
    Rank novel gene candidates from RWR scores.

    Filters out pathway relay nodes (PW: prefix), drug nodes (DRUG: prefix),
    and genes already known to be associated with the disease.

    Args:
        rwr_scores:   Stationary distribution from run_rwr.
        known_genes:  Genes to exclude (seeds + prior known associations).
        top_k:        Number of top candidates to return (default 20).

    Returns:
        List of {gene, brg_score, novel} dicts sorted descending by brg_score.
    """
    known_set = set(known_genes)
    candidates = [
        (node, score)
        for node, score in rwr_scores.items()
        if not node.startswith("PW:")
        and not node.startswith("DRUG:")
        and node not in known_set
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [
        {"gene": node, "brg_score": round(score, 6), "novel": True}
        for node, score in candidates[:top_k]
    ]


def run(
    causal_discovery_result: dict,
    kg_completion_result: dict,
    disease_query: str,
) -> dict:
    """BRG diffusion for novel gene-disease link prediction seeded from causal discovery."""
    warnings: list[str] = []

    # --- Seed genes from causal discovery top_genes weighted by |ota_gamma| ---
    top_genes: list[dict] = causal_discovery_result.get("top_genes", [])
    seed_scores: dict[str, float] = {
        r["gene"]: abs(r.get("ota_gamma", 0.0))
        for r in top_genes
        if abs(r.get("ota_gamma", 0.0)) > 0
    }

    if not seed_scores and top_genes:
        # Fallback: uniform weights when all gammas are zero
        seed_scores = {r["gene"]: 1.0 for r in top_genes if r.get("gene")}
        warnings.append("All ota_gamma values are 0; using uniform seed weights.")

    gene_names = list(seed_scores.keys())

    # --- PPI edges from STRING via MCP server ---
    ppi_edges: list[dict] = []
    try:
        from mcp_servers.pathways_kg_server import get_string_interactions
        ppi_result = get_string_interactions(gene_names[:8], min_score=700)
        ppi_edges = ppi_result.get("interactions", [])
    except Exception as exc:
        warnings.append(f"STRING PPI fetch failed: {exc}")

    # --- Drug-target pairs from KG completion result ---
    drug_target_pairs: list[dict] = []
    try:
        drug_target_pairs = kg_completion_result.get("drug_target_summary", [])
    except Exception as exc:
        warnings.append(f"drug_target_summary extraction failed: {exc}")

    # --- Pathway-gene map from KG completion result ---
    pathway_gene_map: dict[str, list[str]] = {}
    try:
        pathway_gene_map = kg_completion_result.get("pathway_gene_map", {})
    except Exception as exc:
        warnings.append(f"pathway_gene_map extraction failed: {exc}")

    # --- Build BRG ---
    adj: dict[str, list[tuple[str, float]]] = {}
    try:
        known_disease_genes = gene_names
        adj = build_brg(ppi_edges, pathway_gene_map, drug_target_pairs, known_disease_genes)
    except Exception as exc:
        warnings.append(f"build_brg failed: {exc}")
        adj = {}

    # --- Run RWR ---
    rwr_scores: dict[str, float] = {}
    try:
        if seed_scores:
            rwr_scores = run_rwr(adj, seed_scores)
        else:
            warnings.append("No seed genes available; skipping RWR.")
    except Exception as exc:
        warnings.append(f"run_rwr failed: {exc}")

    # --- Score novel links ---
    novel_candidates: list[dict] = []
    try:
        novel_candidates = score_novel_links(rwr_scores, gene_names)
    except Exception as exc:
        warnings.append(f"score_novel_links failed: {exc}")

    n_brg_edges = sum(len(neighbors) for neighbors in adj.values()) // 2

    return {
        "novel_candidates":   novel_candidates,
        "n_seed_genes":       len(seed_scores),
        "n_brg_edges":        n_brg_edges,
        "n_novel_candidates": len(novel_candidates),
        "warnings":           warnings,
    }
