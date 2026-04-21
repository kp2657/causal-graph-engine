"""
pipelines/synthetic_programs.py — Discovery and use of synthetic pathway programs.

A SyntheticProgram is a virtual Perturb-seq program built from:
  - Pathway database (Reactome) + GWAS enrichment
  - Used when a gene has no Perturb-seq β (n_programs=0)

Genes that don't appear in Perturb-seq screens (CFH, C3, CFB, complement genes,
structural ECM genes) get β=0 in the default pipeline.  Synthetic programs give
these "orphan genes" a β by mapping them to disease-enriched Reactome pathways.

γ formula:
    gamma_raw = -log10(fdr) × mean_ot_genetic_score_of_gwas_members
    gamma = clip(gamma_raw / 5.0, 0.05, 0.85)

    Rationale: fdr 0.001 → -log10 = 3.0; mean OT = 0.5 → gamma = 3.0*0.5/5 = 0.30

β formula (for a gene IN the pathway):
    effective_loading = pqtl_beta if available, else ot_score if available, else 0.3
    sign = sign(pqtl_beta) if available, else +1
    beta = sign × |effective_loading|
    sigma = 0.45

Public API:
    discover_synthetic_programs(disease, gwas_genes, ot_genetic_scores, n_programs=15, fdr_threshold=0.1)
        -> list[SyntheticProgram]

    compute_synthetic_beta(gene, programs, loading=1.0, pqtl_beta=None, ot_score=0.0)
        -> dict | None   {beta, sigma, tier, program_name, source, gamma_program}

    SyntheticProgramRegistry   — caches per disease (session-scoped)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class SyntheticProgram:
    """A virtual Perturb-seq program built from Reactome pathway enrichment."""
    name: str                          # "Complement cascade"
    pathway_id: str                    # "R-HSA-166658"
    gwas_genes_in_pathway: list[str]   # GWAS genes that are members
    gamma: float                       # disease relevance
    gamma_source: str                  # "reactome_fdr_ot_weighted"
    pvalue: float                      # Reactome enrichment p-value
    fdr: float                         # FDR
    n_gwas_hits: int                   # number of GWAS genes in pathway
    n_pathway_total: int               # total genes in pathway
    disease: str                       # disease this was built for


def _compute_gamma(fdr: float, mean_ot_score: float) -> float:
    """
    Compute program-disease γ from enrichment FDR and mean OT genetic scores.

    gamma_raw = -log10(fdr) × mean_ot_score
    gamma = clip(gamma_raw / 5.0, 0.05, 0.85)
    """
    fdr_safe = max(fdr, 1e-10)
    gamma_raw = (-math.log10(fdr_safe)) * max(mean_ot_score, 0.05)
    return min(gamma_raw / 5.0, 0.85)


def _programs_to_cache(programs: list[SyntheticProgram]) -> list[dict]:
    return [asdict(p) for p in programs]


def _programs_from_cache(rows: list[dict]) -> list[SyntheticProgram]:
    return [SyntheticProgram(**r) for r in rows]


def discover_synthetic_programs(
    disease: str,
    gwas_genes: list[str],
    ot_genetic_scores: dict[str, float],
    n_programs: int = 15,
    fdr_threshold: float = 0.1,
) -> list[SyntheticProgram]:
    """
    Discover synthetic programs from Reactome pathway enrichment.

    Algorithm:
        1. POST gwas_genes to Reactome enrichment
        2. Filter pathways by FDR
        3. For each enriched pathway, build SyntheticProgram with γ derived
           from enrichment FDR + mean OT score of pathway GWAS members

    Args:
        disease:           Disease name (e.g. "AMD")
        gwas_genes:        GWAS-significant gene symbols
        ot_genetic_scores: OT genetic scores per gene {gene: score}
        n_programs:        Maximum number of programs to return
        fdr_threshold:     Maximum FDR for pathway inclusion

    Returns:
        List of SyntheticProgram sorted by gamma descending
    """
    from mcp_servers.reactome_server import get_enriched_pathways, get_pathway_members_from_token

    if not gwas_genes:
        return []

    result = get_enriched_pathways(gwas_genes)

    if result.get("error") and not result.get("pathways"):
        return []

    token = result.get("token")
    all_pathways = result.get("pathways", [])

    # Filter by FDR
    enriched = [p for p in all_pathways if p["fdr"] <= fdr_threshold]

    programs: list[SyntheticProgram] = []

    for pw in enriched[:n_programs]:
        # Get which GWAS genes are in this pathway
        if token:
            members = get_pathway_members_from_token(token, pw["stId"])
        else:
            members = []

        # If Reactome didn't give us members, fall back to n_found count
        if not members:
            members = gwas_genes[:pw.get("n_found", 0)]

        # Compute gamma
        member_ot_scores = [ot_genetic_scores.get(g, 0.0) for g in members]
        mean_ot = sum(member_ot_scores) / max(len(member_ot_scores), 1)
        gamma = _compute_gamma(pw["fdr"], mean_ot)

        programs.append(SyntheticProgram(
            name=pw["name"],
            pathway_id=pw["stId"],
            gwas_genes_in_pathway=members,
            gamma=gamma,
            gamma_source="reactome_fdr_ot_weighted",
            pvalue=pw["pvalue"],
            fdr=pw["fdr"],
            n_gwas_hits=len(members),
            n_pathway_total=pw["n_total"],
            disease=disease,
        ))

    # Sort by gamma descending
    programs.sort(key=lambda p: p.gamma, reverse=True)
    return programs


def compute_synthetic_beta(
    gene: str,
    programs: list[SyntheticProgram],
    loading: float = 1.0,
    pqtl_beta: float | None = None,
    ot_score: float = 0.0,
) -> dict | None:
    """
    Compute synthetic β for a gene from pathway-program membership.

    Steps:
        1. Look up gene in Reactome to find its pathway memberships
        2. Find matching synthetic programs
        3. Use best (highest gamma) match
        4. Compute β from pQTL/OT score loading, sign from pQTL or default +1

    Args:
        gene:      Gene symbol (e.g. "CFH")
        programs:  List of SyntheticProgram for this disease
        loading:   Default loading if no instrument available
        pqtl_beta: pQTL β (provides sign + magnitude)
        ot_score:  OT genetic score (used as loading if pqtl_beta is None)

    Returns:
        {beta, sigma, tier, program_name, pathway_id, program_gamma, n_gwas_hits, source}
        or None if gene is not in any enriched pathway.
    """
    from mcp_servers.reactome_server import get_gene_pathways

    # Get all Reactome pathways for this gene
    gene_pathways = get_gene_pathways(gene)
    gene_pathway_ids = {p["stId"] for p in gene_pathways}

    # Also check gwas_genes_in_pathway membership (token-based member list)
    matches = []
    for prog in programs:
        in_pathway = (
            prog.pathway_id in gene_pathway_ids
            or gene in prog.gwas_genes_in_pathway
        )
        if in_pathway:
            matches.append(prog)

    if not matches:
        return None

    # Best = highest gamma
    best = max(matches, key=lambda p: p.gamma)

    # Compute effective loading and sign
    if pqtl_beta is not None:
        sign = math.copysign(1.0, pqtl_beta)
        effective_loading = abs(pqtl_beta)
    elif ot_score > 0:
        sign = 1.0
        effective_loading = ot_score
    else:
        sign = 1.0
        effective_loading = 0.3

    beta = sign * effective_loading

    return {
        "beta": beta,
        "sigma": 0.45,
        "tier": "Tier2s_SyntheticPathway",
        "program_name": best.name,
        "pathway_id": best.pathway_id,
        "program_gamma": best.gamma,
        "n_gwas_hits": best.n_gwas_hits,
        "source": f"Reactome pathway enrichment (FDR={best.fdr:.3f})",
        # Keys to match estimate_beta() output shape
        "beta_sigma": 0.45,
        "evidence_tier": "Tier2s_SyntheticPathway",
        "data_source": f"Reactome:{best.pathway_id}_{best.name[:40]}",
        "note": (
            f"Synthetic pathway program '{best.name}' "
            f"(FDR={best.fdr:.3e}, γ={best.gamma:.3f}, "
            f"{best.n_gwas_hits} GWAS genes in pathway)"
        ),
    }


_REACTOME_CACHE_DIR = Path(__file__).parent.parent / "data" / "reactome_cache"
_REACTOME_CACHE_TTL_DAYS = 30


def _disease_cache_path(disease: str) -> "Path":
    return _REACTOME_CACHE_DIR / f"{disease}_programs.json"


def _load_disk_cache(disease: str, ot_scores: dict[str, float]) -> list[SyntheticProgram] | None:
    """Load programs from disk cache, reweighting gamma with current OT scores.
    Returns None if cache is absent, expired, or corrupt."""
    import json, time
    from pathlib import Path

    path = _disease_cache_path(disease)
    if not path.exists():
        return None
    age_days = (time.time() - path.stat().st_mtime) / 86400
    if age_days > _REACTOME_CACHE_TTL_DAYS:
        return None
    try:
        rows = json.loads(path.read_text())
        programs = _programs_from_cache(rows)
        # Reweight gamma with caller's current OT scores
        for p in programs:
            member_scores = [ot_scores.get(g, 0.0) for g in p.gwas_genes_in_pathway]
            mean_ot = sum(member_scores) / max(len(member_scores), 1)
            p.gamma = _compute_gamma(p.fdr, mean_ot)
        programs.sort(key=lambda p: p.gamma, reverse=True)
        return programs
    except Exception:
        return None


def _save_disk_cache(disease: str, programs: list[SyntheticProgram]) -> None:
    """Persist programs to disk. Silent on failure."""
    import json
    try:
        _REACTOME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _disease_cache_path(disease).write_text(
            json.dumps(_programs_to_cache(programs), indent=2)
        )
    except Exception:
        pass


class SyntheticProgramRegistry:
    """
    Two-level cache for SyntheticProgram lists per disease:
      1. In-memory session cache (_cache) — zero-cost within a pipeline run
      2. On-disk JSON cache (data/reactome_cache/{disease}_programs.json) —
         survives pipeline restarts, avoiding Reactome 504s on repeat runs.
         TTL = 30 days.  clear() removes both levels.

    Usage:
        programs = SyntheticProgramRegistry.get_or_build(
            "AMD", gwas_genes, ot_scores
        )
    """
    _cache: dict[str, list[SyntheticProgram]] = {}

    @classmethod
    def get_or_build(
        cls,
        disease: str,
        gwas_genes: list[str],
        ot_scores: dict[str, float],
        **kwargs: Any,
    ) -> list[SyntheticProgram]:
        """
        Return cached programs for disease, or build them if not cached.

        Check order: session cache → disk cache → Reactome API.
        Only writes disk cache on successful API result.

        Args:
            disease:    Disease name (cache key)
            gwas_genes: GWAS gene list (used for enrichment if not cached)
            ot_scores:  OT genetic scores per gene
            **kwargs:   Forwarded to discover_synthetic_programs

        Returns:
            List of SyntheticProgram (empty if Reactome fails and no cache)
        """
        # 1. Session cache (fastest)
        if disease in cls._cache and cls._cache[disease]:
            return cls._cache[disease]

        # 2. Disk cache (survives restarts, avoids 504 on re-runs)
        disk = _load_disk_cache(disease, ot_scores)
        if disk:
            cls._cache[disease] = disk
            return disk

        # 3. Live Reactome call
        programs = discover_synthetic_programs(disease, gwas_genes, ot_scores, **kwargs)
        if programs:
            cls._cache[disease] = programs
            _save_disk_cache(disease, programs)
        return programs

    @classmethod
    def clear(cls, disease: str | None = None) -> None:
        """Clear session + disk cache for one disease or all diseases."""
        if disease is not None:
            cls._cache.pop(disease, None)
            try:
                _disease_cache_path(disease).unlink(missing_ok=True)
            except Exception:
                pass
        else:
            cls._cache.clear()
            try:
                for f in _REACTOME_CACHE_DIR.glob("*_programs.json"):
                    f.unlink(missing_ok=True)
            except Exception:
                pass

    @classmethod
    def is_cached(cls, disease: str) -> bool:
        """Check if programs are in session cache for a disease."""
        return disease in cls._cache
