"""
agents/tier5_writer/literature_validation_agent.py — Phase Q: Literature Integration.

Validates the top-ranked targets from the causal pipeline against the published
literature using PubMed / Europe PMC.

For each top target gene (up to 5), this agent:
  1. Searches PubMed for papers linking the gene to the disease
  2. Classifies papers as supporting or contradicting using title heuristics
  3. Computes a temporal decay factor: older papers carry less weight
       >10yr ago → 0.6×   |   >5yr → 0.8×   |   ≤5yr → 1.0×
  4. Assigns literature confidence: SUPPORTED / MODERATE / NOVEL / CONTRADICTED

Returns:
  literature_evidence: dict[gene → LitEvidenceRecord]
  n_genes_searched, n_genes_supported, n_genes_novel, n_genes_contradicted
  literature_summary: one-paragraph narrative

Local mode: deterministic Python (MCP functions called directly; no Claude API).
SDK mode:   Claude reads abstracts, classifies supporting/contradicting more precisely.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Current year for temporal decay
_REFERENCE_YEAR = 2026

# Minimum papers to classify as SUPPORTED
_SUPPORTED_THRESHOLD = 5
_MODERATE_THRESHOLD  = 1

# Title keywords for simple local-mode classification
_SUPPORTING_KEYWORDS  = {
    "causal", "therapeutic", "treatment", "therapy", "inhibitor", "inhibition",
    "target", "therapeutic target", "association", "risk factor", "promotes",
    "drives", "activates", "regulates", "required", "essential",
}
_CONTRADICTING_KEYWORDS = {
    "no association", "no significant", "not associated", "not a risk",
    "refuted", "failed", "null result", "no effect", "no evidence",
    "negative", "not required",
}


def _age_weight(year_str: str) -> float:
    """Return temporal decay weight based on paper publication year."""
    try:
        age = _REFERENCE_YEAR - int(str(year_str)[:4])
    except (ValueError, TypeError):
        return 0.9  # unknown year: moderate discount
    if age > 10:
        return 0.6
    if age > 5:
        return 0.8
    return 1.0


def _classify_title(title: str) -> str:
    """Return 'supporting' or 'contradicting' based on title keywords."""
    t = title.lower()
    if any(kw in t for kw in _CONTRADICTING_KEYWORDS):
        return "contradicting"
    return "supporting"


def _search_single_gene(gene: str, disease: str) -> dict:
    """
    Call the literature MCP directly and parse results for one gene.

    Returns a LitEvidenceRecord dict.
    """
    from mcp_servers.literature_server import search_gene_disease_literature

    search_query = f'"{gene}" AND "{disease}"'
    try:
        result = search_gene_disease_literature(gene, disease, max_results=10)
    except Exception as exc:
        return {
            "n_papers_found":        0,
            "n_supporting":          0,
            "n_contradicting":       0,
            "key_citations":         [],
            "recency_score":         1.0,
            "temporal_decay_factor": 1.0,
            "literature_confidence": "NOVEL",
            "search_query":          search_query,
            "error":                 str(exc),
        }

    articles = result.get("articles", [])
    total_found = result.get("total_found", len(articles))

    n_supporting    = 0
    n_contradicting = 0
    age_weights: list[float] = []
    key_citations: list[dict] = []

    for art in articles:
        if "error" in art:
            continue
        classification = _classify_title(art.get("title", ""))
        weight         = _age_weight(art.get("year", ""))
        age_weights.append(weight)

        if classification == "contradicting":
            n_contradicting += 1
        else:
            n_supporting += 1

        if len(key_citations) < 3:
            key_citations.append({
                "pmid":           art.get("pmid", ""),
                "title":          art.get("title", ""),
                "authors":        art.get("authors", ""),
                "year":           art.get("year", ""),
                "journal":        art.get("journal", ""),
                "classification": classification,
            })

    recency = sum(age_weights) / len(age_weights) if age_weights else 1.0

    # Literature confidence
    if n_contradicting >= 2 and n_contradicting > n_supporting:
        confidence = "CONTRADICTED"
    elif n_supporting >= _SUPPORTED_THRESHOLD:
        confidence = "SUPPORTED"
    elif n_supporting >= _MODERATE_THRESHOLD:
        confidence = "MODERATE"
    else:
        confidence = "NOVEL"

    return {
        "n_papers_found":        total_found,
        "n_supporting":          n_supporting,
        "n_contradicting":       n_contradicting,
        "key_citations":         key_citations,
        "recency_score":         round(recency, 3),
        "temporal_decay_factor": round(recency, 3),
        "literature_confidence": confidence,
        "search_query":          search_query,
    }


def _build_summary(lit_evidence: dict[str, dict], disease: str) -> str:
    """Build a one-paragraph literature summary for the PI report."""
    supported    = [g for g, e in lit_evidence.items() if e["literature_confidence"] == "SUPPORTED"]
    moderate     = [g for g, e in lit_evidence.items() if e["literature_confidence"] == "MODERATE"]
    novel        = [g for g, e in lit_evidence.items() if e["literature_confidence"] == "NOVEL"]
    contradicted = [g for g, e in lit_evidence.items() if e["literature_confidence"] == "CONTRADICTED"]

    parts: list[str] = []
    if supported:
        parts.append(
            f"{', '.join(supported)} {'has' if len(supported) == 1 else 'have'} "
            f"strong published evidence in {disease} (≥{_SUPPORTED_THRESHOLD} PubMed papers each)."
        )
    if moderate:
        parts.append(
            f"{', '.join(moderate)} {'has' if len(moderate) == 1 else 'have'} "
            f"moderate published evidence ({_MODERATE_THRESHOLD}–{_SUPPORTED_THRESHOLD-1} papers)."
        )
    if novel:
        parts.append(
            f"{', '.join(novel)} {'is a novel target' if len(novel) == 1 else 'are novel targets'} "
            f"with no direct {disease} literature — these may represent new discovery opportunities."
        )
    if contradicted:
        parts.append(
            f"CAUTION: {', '.join(contradicted)} {'has' if len(contradicted) == 1 else 'have'} "
            f"contradicting literature — verify causal evidence before prioritising."
        )

    if not parts:
        return f"Literature search completed for {len(lit_evidence)} targets in {disease}."
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(prioritization_result: dict, disease_query: dict) -> dict:
    """
    Run literature validation for top-ranked targets.

    Args:
        prioritization_result: Output of target_prioritization_agent.
        disease_query:         Disease context dict with disease_name.

    Returns:
        Structured literature evidence per gene.
    """
    targets = sorted(
        prioritization_result.get("targets", []),
        key=lambda t: t.get("rank", 99),
    )[:5]

    disease      = disease_query.get("disease_name", "")
    lit_evidence: dict[str, dict] = {}

    for rec in targets:
        gene = rec.get("target_gene", "")
        if not gene:
            continue
        lit_evidence[gene] = _search_single_gene(gene, disease)

    n_supported    = sum(1 for e in lit_evidence.values() if e["literature_confidence"] == "SUPPORTED")
    n_novel        = sum(1 for e in lit_evidence.values() if e["literature_confidence"] == "NOVEL")
    n_contradicted = sum(1 for e in lit_evidence.values() if e["literature_confidence"] == "CONTRADICTED")

    return {
        "literature_evidence":  lit_evidence,
        "n_genes_searched":     len(lit_evidence),
        "n_genes_supported":    n_supported,
        "n_genes_novel":        n_novel,
        "n_genes_contradicted": n_contradicted,
        "literature_summary":   _build_summary(lit_evidence, disease),
    }
