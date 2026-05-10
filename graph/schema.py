"""
BioCypher + BioLink-compliant graph schema definitions.
Defines node types, edge types, and their required/optional properties.
Used by db.py to initialize Kùzu tables and by ingestion.py to validate edges.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Node type definitions
# ---------------------------------------------------------------------------

NODE_TYPES: dict[str, dict] = {
    "Gene": {
        "biolink_class": "Gene",
        "id_prefix": "ENSEMBL",
        "properties": {
            "id": "STRING",             # ENSEMBL ID or HGNC symbol
            "symbol": "STRING",
            "entrez_id": "STRING",
            "ensembl_id": "STRING",
            "gnomad_pli": "DOUBLE",
            "gnomad_loeuf": "DOUBLE",
            "is_chip_driver": "BOOLEAN",
            "is_viral_gene": "BOOLEAN",
        },
        "primary_key": "id",
    },
    "CellularProgram": {
        "biolink_class": "BiologicalProcess",
        "id_prefix": "CGE",             # causal-graph-engine custom prefix
        "properties": {
            "id": "STRING",
            "name": "STRING",
            "cell_type": "STRING",
            "top_genes": "STRING",      # JSON-encoded list (Kùzu has no native list type in all versions)
            "reactome_pathways": "STRING",
            "perturb_seq_source": "STRING",
            "cnmf_k": "INT64",
        },
        "primary_key": "id",
    },
    "DiseaseTrait": {
        "biolink_class": "Disease",
        "id_prefix": "EFO",
        "properties": {
            "id": "STRING",             # EFO ID preferred, e.g. EFO_0001645
            "name": "STRING",
            "efo_id": "STRING",
            "icd10_codes": "STRING",    # JSON-encoded list
            "h2_estimate": "DOUBLE",
        },
        "primary_key": "id",
    },
    "Drug": {
        "biolink_class": "ChemicalEntity",
        "id_prefix": "CHEMBL",
        "properties": {
            "id": "STRING",             # ChEMBL ID
            "name": "STRING",
            "chembl_id": "STRING",
            "mechanism_of_action": "STRING",
            "modality": "STRING",       # small_molecule | biologic | cell_therapy
        },
        "primary_key": "id",
    },
    "Virus": {
        "biolink_class": "OrganismalEntity",
        "id_prefix": "NCBITaxon",
        "properties": {
            "id": "STRING",             # NCBI Taxonomy ID
            "name": "STRING",
            "ncbi_taxon_id": "STRING",
        },
        "primary_key": "id",
    },
}


# ---------------------------------------------------------------------------
# Edge type definitions
# ---------------------------------------------------------------------------
# Each entry defines which Kùzu relationship table to use and required fields.

EDGE_TYPES: dict[str, dict] = {
    "RegulatesProgram": {
        "biolink_predicate": "regulates",
        "from_node_type": ["Gene", "Drug", "Virus"],
        "to_node_type": ["CellularProgram"],
        "required_fields": [
            "beta",
            "ci_lower", "ci_upper",
            "evidence_type",
            "evidence_tier",
            "data_source",
            "data_source_version",
            "method",
            "cell_type",
            "graph_version",
            "created_at",
        ],
        "optional_fields": [
            "ancestry",
            "e_value",
            "perturb_seq_gene_count",
        ],
    },
    "DrivesTrait": {
        "biolink_predicate": "causally_related_to",
        "from_node_type": ["CellularProgram"],
        "to_node_type": ["DiseaseTrait"],
        "required_fields": [
            "gamma",
            "ci_lower", "ci_upper",
            "method",
            "evidence_tier",
            "data_source",
            "data_source_version",
            "n_modifier_paths",
            "graph_version",
            "created_at",
        ],
        "optional_fields": [
            "validation_sid",
            "validation_shd",
            "mr_ivw",
            "mr_egger_intercept",
            "e_value",
        ],
    },
    "CausesTrait": {
        "biolink_predicate": "causally_related_to",
        "from_node_type": ["Gene", "Drug", "Virus"],
        "to_node_type": ["DiseaseTrait"],
        "required_fields": [
            "effect_size",
            "ci_lower", "ci_upper",
            "path_type",
            "evidence_type",
            "evidence_tier",
            "data_source",
            "data_source_version",
            "graph_version",
            "created_at",
        ],
        "optional_fields": [
            "mr_ivw",
            "mr_egger_intercept",
            "mr_weighted_median",
            "mr_presso_outlier",
            "lof_burden_beta",
            "gwas_pip",
            "e_value",
        ],
    },
}


# ---------------------------------------------------------------------------
# Evidence tier requirements (enforced at ingestion)
# ---------------------------------------------------------------------------

EVIDENCE_TIERS: dict[str, dict] = {
    "Tier1_Interventional": {
        "required": [
            "MR with p<5e-8 and ≥3 instruments",
            "Perturb-seq via inspre/SCONE (FDR<0.05)",
        ],
        "logic": "BOTH required",
        "update_frequency": "On new major Perturb-seq or GWAS release",
        "deprecation": "Never — demoted only if contradicted by equivalent-power intervention",
    },
    "Tier2_Convergent": {
        "required": [
            "GWAS coloc H4 ≥ 0.8",
            "Perturbation support (inspre or Perturb-seq)",
            "Regulatory annotation (ABC/eQTL)",
        ],
        "logic": "TWO of three required",
        "update_frequency": "Quarterly against GWAS Catalog updates",
        "deprecation": "Re-assessed quarterly; requires active support from 2/3 criteria",
    },
    "Tier3_Provisional": {
        "required": [
            "BioPathNet prediction OR PaperQA2 literature synthesis",
        ],
        "logic": "ONE required",
        "update_frequency": "Weekly literature monitoring",
        "deprecation": "Deprecated after 2 years without upgrade to Tier 2",
    },
}


# ---------------------------------------------------------------------------
# Disease → relevant traits for γ estimation
# Keys are the short-form disease IDs used throughout the pipeline.
# Values are the trait labels that PROVISIONAL_GAMMAS and gamma lookup use.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Disease → cell type context for β_{gene→P} estimation
# Determines which Perturb-seq source, GTEx tissue, and LINCS cell line to
# use when estimating β for a given disease.  This is the single config that
# makes the Ota framework generalize beyond K562/blood.
# ---------------------------------------------------------------------------

DISEASE_CELL_TYPE_MAP: dict[str, dict] = {
    "RA": {
        "cell_types":        ["CD4_T_cell"],
        "primary_tissue":    "CD4_T_cell",
        # GTEx has no CD4+ T cell tissue. Whole_Blood is used only as a last-resort
        # fallback when eQTL Catalogue (OneK1K CD4+ T cell) returns nothing for a gene.
        # The sc_eqtl_study / sc_eqtl_cell_types fields below are the primary eQTL source.
        "gtex_tissue":       "Whole_Blood",   # fallback only — not cell-type-matched for RA
        "gtex_tissues_secondary": ["Spleen"],
        "perturb_seq_source": "czi_2025_cd4t_perturb",
        "scperturb_dataset": "czi_2025_cd4t_perturb",
        "lincs_cell_line":   None,
        "pqtl_study_priority": ["Sun2023", "Sun2018", "Ferkingstad2021"],
        "pqtl_key_genes": ["IL6R", "PTPN22", "CTLA4", "CD80", "MS4A1"],
        "sc_eqtl_study":   "OneK1K",          # Yazar 2022 — 982 donors, 14 PBMC cell types
        "sc_eqtl_cell_types": ["CD4_T_cell", "B_cell", "monocyte"],
        "notes": (
            "RA: CD4+ Th1/Th17 synovial inflammation. CZI CD4+ T Perturb-seq (GSE314342) is "
            "the primary β source — genome-scale primary human T cells, Th1/Th17-relevant. "
            "Approved biologics: tocilizumab (IL6R, 2010), rituximab (MS4A1/CD20, 2006 RA), "
            "abatacept (CD80/CTLA4-Ig, 2005). JAK inhibitors: tofacitinib, baricitinib. "
            "Key GWAS loci: PTPN22, HLA-DRB1, STAT4, IL6R, CD28, CTLA4. "
            "CellxGene Census: 23,947 PBMC CD4+ T cells (dataset d18736c3). "
            "Okada 2014 GWAS: ebi-a-GCST002318 (~100k samples)."
        ),
    },
    "CAD": {
        "cell_types":        ["cardiac_endothelial_cell"],
        "primary_tissue":    "whole_blood",
        "gtex_tissue":       "Whole_Blood",   # GTEx tissue ID for eQTL lookup
        # Secondary tissues tried in order when primary has no eQTL for a gene.
        # Liver covers LDLR/PCSK9/HMGCR (hepatic lipid); heart for cardiomyocyte-specific genes.
        "gtex_tissues_secondary": ["Liver", "Heart_Left_Ventricle", "Artery_Coronary"],
        "perturb_seq_source": "schnitzler_cad_vascular",  # 332 CAD GWAS genes in HCASMC/HAEC
        "scperturb_dataset": "schnitzler_cad_vascular",
        "lincs_cell_line":   "VCAP",           # LINCS proxy — best available for metabolic
        # pQTL instruments — key for LPA (no eQTL; driven by kringle repeat number)
        "pqtl_study_priority": ["Sun2023", "Sun2018", "Ferkingstad2021"],
        "pqtl_key_genes": ["LPA", "PCSK9", "APOB", "CRP", "LDLR"],
        # sc-eQTL: monocytes are the disease-relevant cell type for CHIP/inflammation
        "sc_eqtl_study":   "OneK1K",        # OneK1K — 14 PBMC cell types
        "sc_eqtl_cell_types": ["CD14_positive_monocyte", "monocyte", "macrophage"],
        "notes": "CAD vascular biology; Schnitzler 2023 (GSE210681) 332 risk genes in HCASMC/HAEC preferred",
    },
}


# ---------------------------------------------------------------------------
# Available Perturb-seq / scPerturb datasets — single source of truth
# for what's downloadable and which diseases they serve.
# ---------------------------------------------------------------------------

PERTURB_SEQ_SOURCES: dict[str, dict] = {
    "K562": {
        "description":        "Replogle 2022 genome-wide K562 Perturb-seq",
        "cell_type":          "myeloid",
        "cell_line":          "K562",
        "organism":           "human",
        "n_perturbations":    9866,
        "diseases":           ["CAD"],
        "download_url":       "https://ndownloader.figshare.com/files/35773217",  # 357 MB bulk
        "file_size_mb":       357,
        "requires_auth":      False,
        "scperturb_id":       "ReplogleWeissman2022_K562_gwps",
        "status":             "available",   # ready to download
    },
    "K562_essential": {
        "description":        "Replogle 2022 essential gene K562 Perturb-seq",
        "cell_type":          "myeloid",
        "cell_line":          "K562",
        "organism":           "human",
        "n_perturbations":    ~2000,
        "diseases":           ["CAD"],
        "download_url":       "https://ndownloader.figshare.com/files/35780870",  # 76 MB bulk
        "file_size_mb":       76,
        "requires_auth":      False,
        "scperturb_id":       "ReplogleWeissman2022_K562_essential",
        "status":             "available",
    },
    "Schnitzler2023_CAD_vascular": {
        "description":        "Schnitzler 2023 comprehensive Perturb-seq of 332 CAD GWAS risk genes in HCASMC and HAEC",
        "cell_type":          "vascular_smooth_muscle_endothelium",
        "cell_line":          "HCASMC_HAEC",
        "organism":           "human",
        "n_perturbations":    332,
        "diseases":           ["CAD"],
        "geo":                "GSE210681",
        "requires_auth":      False,
        "perturbseq_server_id": "schnitzler_cad_vascular",
        "status":             "needs_download",  # download h5ad from GEO GSE210681
        "download_note":      "Download from GEO GSE210681. Then: python -m mcp_servers.perturbseq_server preprocess schnitzler_cad_vascular <path_to_h5ad>",
    },
    "Frangieh2021_melanoma": {
        "description":        "Frangieh 2021 co-culture Perturb-seq in melanoma + T cells",
        "cell_type":          "melanoma",
        "cell_line":          "melanoma_mixed",
        "organism":           "human",
        "n_perturbations":    248,
        "diseases":           [],
        "download_url":       "https://zenodo.org/record/7041690",
        "file_size_mb":       ~300,
        "requires_auth":      False,
        "scperturb_id":       "FrangiehMacosko2021_co_culture",
        "status":             "available",
    },
    "draeger_2022_ipsc_microglia": {
        "description":        "Dräger 2022 CRISPRi/a druggable genome screen in iPSC-derived microglia",
        "cell_type":          "microglia",
        "cell_line":          "iPSC_microglia",
        "organism":           "human",
        "n_perturbations":    2325,
        "diseases":           [],
        "geo":                "GSE178317",
        "paper_doi":          "10.1038/s41593-022-01131-4",
        "requires_auth":      False,
        "perturbseq_server_id": "draeger_2022_ipsc_microglia",
        "status":             "needs_download",
        "download_note":      "Download from GEO GSE178317. Then: python -m mcp_servers.perturbseq_server preprocess draeger_2022_ipsc_microglia <path_to_h5ad>",
    },
    "czi_2025_cd4t_perturb": {
        "description":        "CZI/Marson 2025 genome-scale Perturb-seq in primary human CD4+ T cells",
        "cell_type":          "CD4_T_cell",
        "cell_line":          "primary_CD4_T",
        "organism":           "human",
        "n_perturbations":    19000,
        "diseases":           ["RA"],
        "geo":                "GSE314342",
        "s3_bucket":          "s3://genome-scale-tcell-perturb-seq/marson2025_data/",
        "paper_doi":          "10.1101/2025.12.23.696273",
        "requires_auth":      False,
        "perturbseq_server_id": "czi_2025_cd4t_perturb",
        "status":             "needs_download",
        "download_note":      "Download from GEO GSE314342 or AWS S3 (open). Then: python -m mcp_servers.perturbseq_server preprocess czi_2025_cd4t_perturb <path_to_h5ad>",
    },
}


# ---------------------------------------------------------------------------
# β tier labels — explicit documentation of what each tier means
# Used in evidence_tier_per_gene annotations throughout the pipeline.
# ---------------------------------------------------------------------------

BETA_TIER_DESCRIPTIONS: dict[str, str] = {
    "Tier1_Interventional": (
        "Direct cell-type-matched Perturb-seq measurement. "
        "Causal basis: physical perturbation (CRISPR KO/KD)."
    ),
    "Tier2_Convergent": (
        "eQTL-MR: genetic instruments (cis-eQTLs) used as instruments for gene expression "
        "in tissue-matched GTEx data; effect projected onto program gene loadings. "
        "Causal basis: Mendelian randomization (genetic randomization of gene expression)."
    ),
    "Tier3_Provisional": (
        "LINCS L1000 genetic perturbation signature: shRNA/ORF perturbation of gene X "
        "in a cell line that may not match the disease-relevant cell type. "
        "Causal basis: direct perturbation, but cell type may be mismatched."
    ),
    "provisional_virtual": (
        "In silico prediction only. Either Geneformer/GEARS virtual perturbation "
        "(extrapolated from perturbation training data) or pathway membership proxy "
        "(no causal basis — annotation only). Must not be used for clinical translation."
    ),
}


DISEASE_TRAIT_MAP: dict[str, list[str]] = {
    "CAD": ["CAD", "LDL-C", "CRP"],
    "RA":  ["RA", "rheumatoid arthritis"],
}


# ---------------------------------------------------------------------------
# Disease name normalizer — maps full disease names to short keys.
# Used throughout the pipeline for disease-key lookups.
# ---------------------------------------------------------------------------

_DISEASE_SHORT_NAMES_FOR_ANCHORS: dict[str, str] = {
    "coronary artery disease":           "CAD",
    "ischemic heart disease":            "CAD",
    "myocardial infarction":             "CAD",
    "heart disease":                     "CAD",
    "rheumatoid arthritis":              "RA",
    "ra":                                "RA",
    "seropositive rheumatoid arthritis": "RA",
    "age-related macular degeneration":  "AMD",
    "macular degeneration":              "AMD",
}
