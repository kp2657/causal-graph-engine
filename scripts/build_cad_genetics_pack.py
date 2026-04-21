from __future__ import annotations

"""
Build a reproducible "CAD genetics pack" artifact for the intended scientific flow.

Contents:
  - OpenGWAS clumped lead variants (requires OPENGWAS_JWT)
  - Open Targets Genetics / Platform: credible sets + L2G (locus→gene)
  - eQTLs matching CAD context (GTEx artery/liver + eQTL Catalogue immune sc-eQTL)
  - Rare-variant burden channel (FinnGen R12 live + UKB WES burden proxy)

Output:
  causal-graph-engine/data/genetics_packs/cad_genetics_pack.json

Run:
  PYTHONPATH="causal-graph-engine" conda run -n causal-graph \
    python -u causal-graph-engine/scripts/build_cad_genetics_pack.py

Env knobs:
  OPENGWAS_JWT=<token>     # enables live OpenGWAS /tophits
  CAD_GWAS_ID=ieu-b-4816   # default CAD GWAS study id (OpenGWAS)
  MAX_TOPHITS=200          # cap tophits
  MAX_GENES=200            # how many genes to annotate with eQTL/burden
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from mcp_servers.gwas_genetics_server import (
    get_opengwas_tophits,
    get_open_targets_genetics_credible_sets,
    get_open_targets_gwas_studies_for_efo,
    get_l2g_scores,
    query_gtex_eqtl,
    get_finngen_burden_results,
)
from mcp_servers.open_targets_server import get_open_targets_disease_targets
from mcp_servers.eqtl_catalogue_server import (
    get_best_sc_eqtl_for_gene,
    get_best_pqtl_for_gene,
)
from mcp_servers.ukb_wes_server import get_burden_direction_for_gene


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    disease_name = "coronary artery disease"
    efo_id = "EFO_0001645"
    cad_gwas_id = os.getenv("CAD_GWAS_ID", "ieu-b-4816").strip()
    max_tophits = int(os.getenv("MAX_TOPHITS", "200"))
    max_genes = int(os.getenv("MAX_GENES", "200"))
    max_ot_studies_used = int(os.getenv("MAX_OT_STUDIES_USED", "3"))
    fast = os.getenv("FAST", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    enable_gtex = os.getenv("ENABLE_GTEX", "1").strip() not in {"0", "false", "FALSE", "no", "NO"}
    enable_sc_eqtl = os.getenv("ENABLE_SC_EQTL", "1").strip() not in {"0", "false", "FALSE", "no", "NO"}
    enable_pqtl = os.getenv("ENABLE_PQTL", "1").strip() not in {"0", "false", "FALSE", "no", "NO"}
    enable_finngen = os.getenv("ENABLE_FINNGEN", "1").strip() not in {"0", "false", "FALSE", "no", "NO"}
    enable_ukb = os.getenv("ENABLE_UKB_WES", "1").strip() not in {"0", "false", "FALSE", "no", "NO"}
    gtex_tissues_env = os.getenv("GTEX_TISSUES", "").strip()
    tissues = (
        [t.strip() for t in gtex_tissues_env.split(",") if t.strip()]
        if gtex_tissues_env
        else ["Artery_Coronary", "Artery_Aorta", "Liver", "Whole_Blood"]
    )

    out_dir = Path("causal-graph-engine/data/genetics_packs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cad_genetics_pack.json"

    # 1) OpenGWAS lead variants (already clumped server-side)
    print(f"[1/5] OpenGWAS tophits for {cad_gwas_id} (requires OPENGWAS_JWT)...", flush=True)
    opengwas = get_opengwas_tophits(cad_gwas_id, pval=5e-8, clump=1, n_max=max_tophits)

    # 2) OT credible sets (trait-level) + L2G (study-level)
    print(f"[2/5] Open Targets credible sets + GWAS studies + L2G for {efo_id}...", flush=True)
    credsets = get_open_targets_genetics_credible_sets(efo_id, min_pip=0.1)
    ot_studies = get_open_targets_gwas_studies_for_efo(efo_id, max_studies=50)
    study_ids = [s["id"] for s in (ot_studies.get("studies") or []) if s.get("id")]

    # Aggregate L2G across OT GWAS studies (keep best score per gene)
    best_l2g: dict[str, dict] = {}
    used_ids = study_ids[: max(1, max_ot_studies_used)]
    for sid in used_ids:
        l2g_part = get_l2g_scores(sid, top_n=50)
        for rec in (l2g_part.get("l2g_genes") or []):
            g = rec.get("gene_symbol")
            s = rec.get("l2g_score")
            if not g or s is None:
                continue
            if g not in best_l2g or float(s) > float(best_l2g[g]["l2g_score"]):
                best_l2g[g] = {**rec, "study_id": sid}

    l2g = {
        "study_ids_used": used_ids,
        "n_studies_available": len(study_ids),
        "l2g_genes": sorted(best_l2g.values(), key=lambda x: x.get("l2g_score", 0), reverse=True),
        "data_source": "OT_Platform_v4_L2G (aggregated across studies)",
    }

    # Candidate genes to annotate:
    # - prioritize L2G genes if available, else fall back to FinnGen burden for a small set later
    l2g_genes = [r.get("gene_symbol") for r in (l2g.get("l2g_genes") or []) if r.get("gene_symbol")]
    genes = l2g_genes[:max_genes]
    if not genes:
        # Fallback: use OT disease targets list (paged + now robust for large sizes)
        print("[2/5] L2G empty; falling back to OT disease targets list...", flush=True)
        ot_targets = get_open_targets_disease_targets(efo_id, max_targets=max_genes, min_overall_score=0.0)
        genes = [t.get("gene_symbol") for t in (ot_targets.get("targets") or []) if t.get("gene_symbol")]

    eqtl: dict[str, dict] = {}
    sc_eqtl: dict[str, dict] = {}
    pqtl: dict[str, dict] = {}
    finngen = {"burden_results": [], "n_found": 0, "note": "skipped"}
    ukb_burden: dict[str, dict] = {}

    if fast:
        print("[FAST=1] Skipping eQTL and burden annotation passes.", flush=True)
    else:
        # 3) eQTLs matching CAD context + sc-eQTL/pQTL fallbacks
        print(f"[3/5] eQTL annotation for {len(genes)} genes (GTEx + sc-eQTL + pQTL)...", flush=True)

        for idx, g in enumerate(genes, start=1):
            if idx % 20 == 0:
                print(f"  - annotated {idx}/{len(genes)} genes...", flush=True)
            # GTEx bulk tissues
            if enable_gtex:
                g_eqtls = []
                for t in tissues:
                    try:
                        r = query_gtex_eqtl(g, t, items_per_page=5)
                        if r.get("eqtls"):
                            g_eqtls.append({"tissue": t, "top": r["eqtls"][0], "n_eqtls": r.get("n_eqtls")})
                    except Exception:
                        pass
                if g_eqtls:
                    eqtl[g] = {"bulk": g_eqtls}

            # sc-eQTL (immune proxy; CAD often macrophage/monocyte)
            if enable_sc_eqtl:
                try:
                    top_sc = get_best_sc_eqtl_for_gene(g, disease="CAD")
                    if top_sc:
                        sc_eqtl[g] = top_sc
                except Exception:
                    pass

            # pQTL (CAD key proteins like LPA/PCSK9)
            if enable_pqtl:
                try:
                    top_p = get_best_pqtl_for_gene(g, disease="CAD")
                    if top_p:
                        pqtl[g] = top_p
                except Exception:
                    pass

        # 4) Rare variant burden channels
        # 4a) FinnGen R12 gene-level burden (fast, public, live)
        if enable_finngen:
            print(f"[4/5] FinnGen burden for {len(genes)} genes...", flush=True)
            finngen = get_finngen_burden_results(disease_name, genes)
        else:
            finngen = {"burden_results": [], "n_found": 0, "note": "skipped"}

        # 4b) UKB WES burden proxy via OT evidence + gnomAD constraint
        if enable_ukb:
            print(f"[5/5] UKB WES burden proxy for {len(genes)} genes...", flush=True)
            for g in genes:
                try:
                    b = get_burden_direction_for_gene(g, disease="CAD")
                    if b:
                        ukb_burden[g] = b
                except Exception:
                    pass

    pack = {
        "disease_name": disease_name,
        "efo_id": efo_id,
        "cad_gwas_id": cad_gwas_id,
        "generated_at": _now_iso(),
        "opengwas_tophits": opengwas,
        "ot_credible_sets": credsets,
        "ot_gwas_studies": ot_studies,
        "ot_l2g": l2g,
        "genes": genes,
        "eqtl": eqtl,
        "sc_eqtl": sc_eqtl,
        "pqtl": pqtl,
        "finngen_burden": finngen,
        "ukb_wes_burden_proxy": ukb_burden,
        "fast_mode": fast,
        "notes": {
            "opengwas": "Requires OPENGWAS_JWT; /tophits returns LD-clumped lead variants (approx clumping).",
            "coloc": "Full coloc requires locus summary stats and LD; OT provides coloc proxies, and TWMR provides ratio-MR when instruments align.",
            "burden": "FinnGen burden is live/public; UKB WES is proxied through OT/constraint signals in this repo.",
        },
    }

    out_path.write_text(json.dumps(pack, indent=2))
    print(f"Wrote {out_path} with {len(genes)} genes.")
    print(f"OpenGWAS hits returned: {pack['opengwas_tophits'].get('n_returned')}")
    print(f"OT credible sets: {len((credsets.get('credible_sets') or []))}")
    print(f"OT L2G genes: {len((l2g.get('l2g_genes') or []))}")


if __name__ == "__main__":
    main()

