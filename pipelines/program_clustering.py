"""
program_clustering.py — Cluster OTA-grounded genes by program influence similarity.

Each gene's causal influence is a vector over programs: [β_{g→P} × γ_{P→trait}]
for each P. Normalising to unit L1 gives a distribution over causal pathways.
Cosine-distance KMeans groups genes that act through the same program mix.

Clusters are meaningful for two reasons:
  - Same cluster = shared mechanism → likely redundant coverage (avoid in combo)
  - Different cluster = complementary mechanism → combination therapy candidates

Only OTA-grounded genes (n_programs_contributing > 0) are clustered.
GWAS-anchored genes receive cluster_id = None.
"""
from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def cluster_by_program_influence(
    top_genes: list[dict],
    max_clusters: int = 8,
    min_genes_per_cluster: int = 2,
) -> list[dict]:
    """
    Annotate each OTA-grounded gene in top_genes with program-cluster fields.

    Args:
        top_genes:              List of gene dicts from causal_discovery_agent.
                                Each must have a "programs" key containing
                                {program_id: contribution} from compute_ota_gamma.
        max_clusters:           Hard cap on k. Actual k = min(max_clusters,
                                n_programs_active, len(grounded)//min_per_cluster).
        min_genes_per_cluster:  Floor that prevents k from exceeding gene count.

    Returns:
        New list with cluster fields injected into each dict:
          - program_cluster_id   int | None
          - dominant_program     str | None
          - cluster_label        str | None   (human-readable)
          - cluster_size         int | None
    """
    import numpy as np

    grounded = [g for g in top_genes if g.get("n_programs_contributing", 0) > 0]
    ungrounded = [g for g in top_genes if g.get("n_programs_contributing", 0) == 0]

    if len(grounded) < 2:
        return _annotate_unclustered(top_genes)

    # Build program universe from all influence vectors
    all_programs: list[str] = []
    for g in grounded:
        progs = g.get("programs") or {}
        if isinstance(progs, dict):
            for p in progs:
                if p not in all_programs:
                    all_programs.append(p)

    if not all_programs:
        return _annotate_unclustered(top_genes)

    prog_idx = {p: i for i, p in enumerate(all_programs)}
    n_progs = len(all_programs)

    # Build influence matrix: rows=genes, cols=programs, values=|contribution|
    mat = np.zeros((len(grounded), n_progs), dtype=float)
    for row, g in enumerate(grounded):
        progs = g.get("programs") or {}
        if isinstance(progs, dict):
            for prog, contrib in progs.items():
                if prog in prog_idx:
                    mat[row, prog_idx[prog]] = abs(float(contrib or 0.0))

    # L1-normalise each row → simplex (distribution over programs)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums

    # Choose k
    k = min(
        max_clusters,
        n_progs,
        max(1, len(grounded) // min_genes_per_cluster),
    )
    k = max(1, k)

    if k == 1:
        # Single cluster — all genes share the same dominant program
        dominant = _dominant_program(mat, all_programs)
        label = _make_label(dominant)
        result = []
        for g in top_genes:
            gc = dict(g)
            if g.get("n_programs_contributing", 0) > 0:
                gc.update({
                    "program_cluster_id": 0,
                    "dominant_program":   dominant,
                    "cluster_label":      label,
                    "cluster_size":       len(grounded),
                })
            else:
                gc.update(_null_cluster_fields())
            result.append(gc)
        return result

    labels, cluster_dominants, cluster_sizes = _kmeans_cosine(mat, k, all_programs)

    # Build annotated output preserving original order
    grounded_iter = iter(zip(grounded, labels))
    grounded_map: dict[str, tuple[int, str, str, int]] = {}
    for g, lbl in zip(grounded, labels):
        dominant = cluster_dominants[lbl]
        grounded_map[g["gene"]] = (
            int(lbl),
            dominant,
            _make_label(dominant),
            cluster_sizes[lbl],
        )

    result = []
    for g in top_genes:
        gc = dict(g)
        if g["gene"] in grounded_map:
            cid, dom, clabel, csize = grounded_map[g["gene"]]
            gc.update({
                "program_cluster_id": cid,
                "dominant_program":   dom,
                "cluster_label":      clabel,
                "cluster_size":       csize,
            })
        else:
            gc.update(_null_cluster_fields())
        result.append(gc)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kmeans_cosine(
    mat: "np.ndarray",
    k: int,
    all_programs: list[str],
    n_init: int = 10,
    max_iter: int = 200,
    random_state: int = 42,
) -> tuple[list[int], dict[int, str], dict[int, int]]:
    """
    KMeans on L2-normalised rows (≡ cosine-distance KMeans).

    Returns:
        labels:           cluster assignment per gene (len = n_genes)
        cluster_dominants: {cluster_id → dominant program name}
        cluster_sizes:    {cluster_id → n_genes}
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    mat_l2 = normalize(mat, norm="l2")

    km = KMeans(
        n_clusters=k,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels_arr = km.fit_predict(mat_l2)
    labels = [int(x) for x in labels_arr]

    # Dominant program per cluster = argmax of centroid in L1-normalised space
    # Use original mat (L1-normalised) for interpretability
    cluster_dominants: dict[int, str] = {}
    cluster_sizes: dict[int, int] = {}
    for cid in range(k):
        members = mat[[i for i, l in enumerate(labels) if l == cid]]
        cluster_sizes[cid] = len(members)
        if len(members) == 0:
            cluster_dominants[cid] = all_programs[0]
            continue
        centroid = members.mean(axis=0)
        dominant_idx = int(centroid.argmax())
        cluster_dominants[cid] = all_programs[dominant_idx]

    return labels, cluster_dominants, cluster_sizes


def _dominant_program(mat: "np.ndarray", all_programs: list[str]) -> str:
    import numpy as np
    centroid = mat.mean(axis=0)
    return all_programs[int(centroid.argmax())]


def _make_label(program_id: str) -> str:
    """Convert program_id to a short human-readable cluster label."""
    return (
        program_id
        .replace("HALLMARK_", "")
        .replace("_", " ")
        .lower()
        .strip()
    )


def _null_cluster_fields() -> dict:
    return {
        "program_cluster_id": None,
        "dominant_program":   None,
        "cluster_label":      None,
        "cluster_size":       None,
    }


def _annotate_unclustered(top_genes: list[dict]) -> list[dict]:
    return [dict(g, **_null_cluster_fields()) for g in top_genes]
