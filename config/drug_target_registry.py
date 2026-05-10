"""
config/drug_target_registry.py — Clinically validated drug targets for pipeline QC.

Convention for expected_gamma_sign:
  -1: gene drives disease (KO is protective) → inhibitor drug target
  +1: gene is protective (KO amplifies disease) → agonist/boost drug target

RA entries: only targets whose primary therapeutic mechanism is in CD4+ T cells,
AND that have a KO in the CZI 11,281-gene perturb-seq library.
Excluded (not in CZI KO library or wrong cell type):
  JAK1    — not in CZI KO library (11,281 targets)
  JAK3    — not in CZI KO library (11,281 targets)
  MS4A1   — CD20, B cell; not expressed in CD4+ T cells
  IL6R    — primary RA mechanism is stromal/fibroblast trans-signaling
  PADI4   — citrullination acts in neutrophils/macrophages
  IRF5    — primary mechanism is myeloid/DC type I IFN production
"""
from __future__ import annotations

# fmt: off
VALIDATED_DRUG_TARGETS: dict[str, list[dict]] = {

    # ── Rheumatoid Arthritis ──────────────────────────────────────────────────
    # CD4+ T cell-intrinsic mechanisms only (CZI Perturb-seq model).
    "RA": [
        dict(gene="TYK2",    drug="deucravacitinib",         status="Phase3_RA/Approved_PsA", expected_sign=-1,
             note="P1104A protective LoF; TYK2 inhibition suppresses IL-23/IFN axis in effector T cells"),
        dict(gene="JAK2",    drug="baricitinib",             status="Approved_RA",            expected_sign=-1,
             note="STAT5 in Tregs and effector T cells; CD4+ T cell KO effect expected"),
        # JAK3: not in CZI KO library (11,281 targets) — excluded from benchmark
        # dict(gene="JAK3", drug="tofacitinib", status="Approved_RA", expected_sign=-1)
        dict(gene="CTLA4",   drug="abatacept",               status="Approved_RA",            expected_sign=+1,
             note="CTLA4-Ig is an agonist of a protective checkpoint; KO amplifies disease (expected γ>0)"),
        dict(gene="IL12RB2", drug="ustekinumab_target_axis", status="Approved_RA",            expected_sign=-1,
             note="IL-12/23 receptor drives Th1/Th17 differentiation in CD4+ T cells"),
        dict(gene="ICOS",    drug="emerging_agonists",       status="Phase2",                 expected_sign=-1,
             note="ICOS costimulation drives effector T cell inflammation; inhibiting ICOS is the drug strategy"),
        dict(gene="TRAF3IP2",drug="sikokitinib",             status="Approved_PsA",           expected_sign=-1,
             note="TRAF3IP2 encodes ACT1; expressed in CD4+ T cells; drives IL-17 signaling in RA/PsA"),
        dict(gene="CD226",   drug="emerging_checkpoint",     status="Preclinical",            expected_sign=-1,
             note="DNAM-1 activating receptor; expressed on CD4+ T cells; drives cytotoxicity in autoimmunity"),
        dict(gene="IL2RA",   drug="basiliximab",             status="Approved_transplant",    expected_sign=-1,
             note="CD25 on CD4+ T cells; drives Treg survival vs. effector activation"),
        dict(gene="PTPN22",  drug="preclinical",             status="Preclinical",            expected_sign=-1,
             note="R620W GOF variant → hyperactivated TCR signaling in CD4+ T cells; drives RA susceptibility"),
        dict(gene="REL",     drug="preclinical",             status="Preclinical",            expected_sign=-1,
             note="c-Rel NF-kB subunit; drives T cell activation and inflammatory gene expression"),
        dict(gene="TRAF1",   drug="GWAS_anchor",             status="GWAS_anchor",            expected_sign=-1,
             note="NF-kB co-activator; expressed in T cells; TRAF1-C5 haplotype associated with RA"),
    ],

    # ── Coronary Artery Disease ───────────────────────────────────────────────
    # Schnitzler Nature 2024 Fig. 2c — 7 genes with clear GWAS colocalization evidence.
    # Schnitzler phenotype: red = KD increases risk (atheroprotective); blue = KD reduces risk (atherogenic driver).
    # expected_sign convention matches Schnitzler KD phenotype (not OTA γ sign):
    #   +1: KD increases risk (gene is atheroprotective; KO amplifies disease)
    #   -1: KD reduces risk (gene is atherogenic driver; KO is protective)
    # PGF excluded: absent from Schnitzler CRISPRi library.
    "CAD": [
        # ── Atheroprotective (red in Schnitzler Fig 2c; KD increases CAD risk) ──
        dict(gene="PLPP3",   drug="Schnitzler2024_Fig2c", status="Research", expected_sign=+1,
             note="Phospholipid phosphatase; atheroprotective; Schnitzler KD→increased risk"),
        dict(gene="NOS3",    drug="Schnitzler2024_Fig2c", status="Research", expected_sign=+1,
             note="eNOS; atheroprotective; Schnitzler KD→increased risk"),
        dict(gene="EXOC3L2", drug="Schnitzler2024_Fig2c", status="Research", expected_sign=+1,
             note="Exocyst complex; atheroprotective; Schnitzler KD→increased risk"),
        dict(gene="CALCRL",  drug="Schnitzler2024_Fig2c", status="Research", expected_sign=+1,
             note="CGRP receptor; atheroprotective; Schnitzler KD→increased risk"),
        # ── Atherogenic drivers (blue in Schnitzler Fig 2c; KD reduces CAD risk) ─
        dict(gene="COL4A1",  drug="Schnitzler2024_Fig2c", status="Research", expected_sign=-1,
             note="Collagen IVα1; atherogenic ECM driver; Schnitzler KD→reduced risk"),
        dict(gene="LOX",     drug="Schnitzler2024_Fig2c", status="Research", expected_sign=-1,
             note="Lysyl oxidase; atherogenic ECM driver; Schnitzler KD→reduced risk"),
        dict(gene="COL4A2",  drug="Schnitzler2024_Fig2c", status="Research", expected_sign=-1,
             note="Collagen IV; atherogenic ECM driver; Schnitzler KD→reduced risk"),
    ],
}
# fmt: on


def get_validated_targets(disease_key: str) -> list[dict]:
    """Return validated drug target list for a disease key (e.g. 'RA', 'CAD')."""
    return VALIDATED_DRUG_TARGETS.get(disease_key.upper(), [])
