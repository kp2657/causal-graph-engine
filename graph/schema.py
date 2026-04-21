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
            "validation_anchor_recovery",
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
    "CAD": {
        "cell_types":        ["myeloid", "hepatocyte", "smooth_muscle"],
        "primary_tissue":    "whole_blood",
        "gtex_tissue":       "Whole_Blood",   # GTEx tissue ID for eQTL lookup
        # Secondary tissues tried in order when primary has no eQTL for a gene.
        # Liver covers LDLR/PCSK9/HMGCR (hepatic lipid); heart for cardiomyocyte-specific genes.
        "gtex_tissues_secondary": ["Liver", "Heart_Left_Ventricle", "Artery_Coronary"],
        "perturb_seq_source": "schnitzler_cad_vascular",  # 332 CAD GWAS genes in HCASMC/HAEC
        "scperturb_dataset": "Schnitzler_GSE210681",
        "lincs_cell_line":   "VCAP",           # LINCS proxy — best available for metabolic
        # pQTL instruments — key for LPA (no eQTL; driven by kringle repeat number)
        "pqtl_study_priority": ["Sun2023", "Sun2018", "Ferkingstad2021"],
        "pqtl_key_genes": ["LPA", "PCSK9", "APOB", "CRP", "LDLR"],
        # sc-eQTL: monocytes are the disease-relevant cell type for CHIP/inflammation
        "sc_eqtl_study":   "OneK1K",        # OneK1K — 14 PBMC cell types
        "sc_eqtl_cell_types": ["CD14_positive_monocyte", "monocyte", "macrophage"],
        "notes": "CAD vascular biology; Schnitzler 2023 (GSE210681) 332 risk genes in HCASMC/HAEC preferred",
    },
    "RA": {
        "cell_types":        ["T_cell", "B_cell", "macrophage", "synoviocyte"],
        "primary_tissue":    "whole_blood",
        "gtex_tissue":       "Whole_Blood",
        "gtex_tissues_secondary": ["Skin_Sun_Exposed_Lower_leg", "Adipose_Subcutaneous"],
        "perturb_seq_source": "scPerturb_PBMC",
        "scperturb_dataset": "PapalexiSatija2021_eccite",  # 111 perturbations, PBMC
        "lincs_cell_line":   "A375",
        "pqtl_study_priority": ["Sun2023", "Sun2018"],
        "pqtl_key_genes": ["IL6", "TNF", "CRP", "IL6R"],
        "sc_eqtl_study":   "OneK1K",
        "sc_eqtl_cell_types": ["CD4_positive_alpha_beta_T_cell", "B_cell", "CD14_positive_monocyte"],
        "notes": "IL-6/JAK-STAT and MHC-II programs; PBMC Perturb-seq (Papalexi 2021) preferred over K562",
    },
    "SLE": {
        "cell_types":        ["plasmacytoid_DC", "B_cell", "T_cell"],
        "primary_tissue":    "whole_blood",
        "gtex_tissue":       "Whole_Blood",
        "gtex_tissues_secondary": ["Kidney_Cortex"],
        "perturb_seq_source": "K562",          # fallback — no PDC Perturb-seq available
        "scperturb_dataset": None,
        "lincs_cell_line":   "A375",
        "notes": "IFN-I pathway central; no PDC Perturb-seq — K562 + eQTL-MR are best available",
    },
    "IBD": {
        "cell_types":        ["colonocyte", "macrophage", "T_cell"],
        "primary_tissue":    "colon",
        "gtex_tissue":       "Colon_Sigmoid",
        # Small intestine captures ileal Crohn's genes (NOD2, IL23R); liver covers systemic inflammation.
        "gtex_tissues_secondary": ["Small_Intestine_Terminal_Ileum", "Colon_Transverse", "Whole_Blood"],
        "perturb_seq_source": "LINCS_HT29",    # HT29 = colon adenocarcinoma, closest available
        "scperturb_dataset": None,
        "lincs_cell_line":   "HT29",
        "pqtl_study_priority": ["Sun2023", "Sun2018"],
        "pqtl_key_genes": ["IL6", "TNF", "CRP", "IL10", "IL23A"],
        "sc_eqtl_study":   "OneK1K",
        "sc_eqtl_cell_types": ["CD14_positive_monocyte", "CD4_positive_alpha_beta_T_cell"],
        "notes": "No gut epithelium Perturb-seq; HT29 (colon) L1000 is the best β source; eQTL-MR in sigmoid colon",
    },
    "AD": {
        "cell_types":        ["microglia", "neuron", "astrocyte"],
        "primary_tissue":    "brain",
        "gtex_tissue":       "Brain_Frontal_Cortex_BA9",
        "gtex_tissues_secondary": ["Brain_Hippocampus", "Brain_Caudate_basal_ganglia", "Whole_Blood"],
        "perturb_seq_source": "scPerturb_iPSC_neuron",
        "scperturb_dataset": "Ursu2022_neurips",  # 96 gene perturbations, iPSC neurons
        "lincs_cell_line":   "A549",           # lung — imperfect; Geneformer preferred for neurons
        "notes": "Microglia absent from all cancer-line Perturb-seq; use Ursu 2022 iPSC + Geneformer virtual",
    },
    "T2D": {
        "cell_types":        ["beta_cell", "hepatocyte", "adipocyte"],
        "primary_tissue":    "pancreas",
        "gtex_tissue":       "Pancreas",
        "gtex_tissues_secondary": ["Liver", "Adipose_Subcutaneous", "Muscle_Skeletal"],
        "perturb_seq_source": "LINCS_HepG2",
        "scperturb_dataset": None,
        "lincs_cell_line":   "HEPG2",          # hepatocytes for metabolic programs
        "notes": "No pancreatic β-cell Perturb-seq; HepG2 for metabolic programs; eQTL-MR in pancreas GTEx",
    },
    "MS": {
        "cell_types":        ["T_cell", "oligodendrocyte", "microglia"],
        "primary_tissue":    "whole_blood",
        "gtex_tissue":       "Whole_Blood",
        "gtex_tissues_secondary": ["Brain_Frontal_Cortex_BA9", "Brain_Caudate_basal_ganglia"],
        "perturb_seq_source": "scPerturb_PBMC",
        "scperturb_dataset": "PapalexiSatija2021_eccite",
        "lincs_cell_line":   "A375",
        "notes": "T cell programs prominent; use PBMC Perturb-seq for immune component",
    },
    "T1D": {
        "cell_types":        ["T_cell", "beta_cell"],
        "primary_tissue":    "whole_blood",
        "gtex_tissue":       "Whole_Blood",
        "gtex_tissues_secondary": ["Pancreas"],
        "perturb_seq_source": "scPerturb_PBMC",
        "scperturb_dataset": "PapalexiSatija2021_eccite",
        "lincs_cell_line":   "A375",
        "notes": "Autoimmune + beta cell destruction; T cell programs via PBMC Perturb-seq",
    },
    "AMD": {
        "cell_types":        ["RPE", "photoreceptor", "Muller_glia"],
        "primary_tissue":    "retina",
        "gtex_tissue":       "Retina",      # GTEx v10 retina tissue
        # Liver is the primary source of complement proteins (CFH, C3, CFD, C5, CFB, CFI).
        # Without liver eQTL, complement targets are systematically underscored despite being
        # the most validated AMD targets (pegcetacoplan approved, danicopan Phase 3, GT005 Phase 2).
        "gtex_tissues_secondary": ["Liver", "Whole_Blood"],
        "perturb_seq_source": "Replogle2022_RPE1",  # RPE1 is retinal epithelial — tissue match
        "scperturb_dataset": "replogle_2022_rpe1",
        "lincs_cell_line":   None,          # no AMD-specific LINCS cell line
        # pQTL instruments — CRITICAL for complement genes with no cis-eQTL
        # CFH Y402H (rs1061170) has plasma pQTL in UKB-PPP/INTERVAL but no GTEx eQTL
        "pqtl_study_priority": ["Sun2023", "Sun2018", "Ferkingstad2021"],
        "pqtl_key_genes": ["CFH", "C3", "CFB", "CFD", "C5", "CFI", "CFHR1", "VEGFA"],
        # No retina-specific sc-eQTL in public eQTL Catalogue; monocyte eQTL for complement regulation
        "sc_eqtl_study":   "OneK1K",
        "sc_eqtl_cell_types": ["CD14_positive_monocyte"],   # complement is expressed in monocytes
        "notes": (
            "RPE + complement biology; Replogle RPE1 for RPE-autonomous genes. "
            "IMPORTANT: CFH/C3/CFD/C5 key GWAS variants are CODING variants (CFH Y402H rs1061170) — "
            "they have NO cis-eQTL in any GTEx tissue. These genes cannot be scored by eQTL-MR. "
            "Correct instrument = pQTL (plasma protein QTL from INTERVAL/ARIC/UKB-PPP). "
            "Liver secondary tissue helps metabolic AMD genes (LIPC gains NES=-0.295 in Liver vs 0 in Retina). "
            "Tier 2p (pQTL-MR) is the primary evidence tier for all complement pathway genes."
        ),
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
        "diseases":           ["CAD", "RA", "SLE"],
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
    "Papalexi2021_PBMC": {
        "description":        "Papalexi 2021 CITE-seq with CRISPR perturbations in PBMCs",
        "cell_type":          "PBMC",
        "cell_line":          "primary_PBMC",
        "organism":           "human",
        "n_perturbations":    111,
        "diseases":           ["RA", "SLE", "MS", "T1D"],
        "download_url":       "https://zenodo.org/record/7041690",  # scPerturb harmonized
        "file_size_mb":       ~200,
        "requires_auth":      False,
        "scperturb_id":       "PapalexiSatija2021_eccite",
        "status":             "available",
    },
    "Ursu2022_iPSC_neuron": {
        "description":        "Ursu 2022 CRISPRi in iPSC-derived neurons (NeurIPS 2022 benchmark)",
        "cell_type":          "iPSC_neuron",
        "cell_line":          "iPSC_derived",
        "organism":           "human",
        "n_perturbations":    96,
        "diseases":           ["AD"],
        "download_url":       "https://zenodo.org/record/7041690",  # scPerturb harmonized
        "file_size_mb":       ~500,
        "requires_auth":      False,
        "scperturb_id":       "Ursu2022_neurips",
        "perturbseq_server_id": "ursu_2022_ipsc_neuron",  # key in perturbseq_server._DATASET_REGISTRY
        "status":             "available",
        "n_genes_note":       "96 genes only — limited coverage; eQTL-MR fills the remaining genome",
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
    "RA":  ["RA", "CRP"],
    "SLE": ["SLE", "CRP"],
    "AD":  ["AD"],
    "IBD": ["IBD", "Crohn_disease", "UC", "CRP"],
    "MS":  ["MS"],
    "T1D": ["T1D"],
    "AMD": ["AMD"],
}


# ---------------------------------------------------------------------------
# Anchor edges — ground-truth benchmark (CAD-relevant subset first)
# ---------------------------------------------------------------------------

ANCHOR_EDGES: list[dict] = [
    # Germline → trait (Tier 1 anchors)
    {"from": "PCSK9",   "to": "LDL-C",              "evidence_type": "germline",    "direction": "negative_lof"},
    {"from": "LDLR",    "to": "LDL-C",              "evidence_type": "germline",    "direction": "negative_lof"},
    {"from": "HMGCR",   "to": "LDL-C",              "evidence_type": "germline",    "direction": "negative_lof"},
    {"from": "FTO",     "to": "BMI",                "evidence_type": "germline",    "direction": "positive"},
    {"from": "HBA1",    "to": "MCH",                "evidence_type": "germline",    "direction": "positive"},
    {"from": "IL6R",    "to": "CRP",                "evidence_type": "germline",    "direction": "negative_lof"},
    {"from": "APOE_e4", "to": "Alzheimer_disease",  "evidence_type": "germline",    "direction": "positive"},
    # Program → trait
    {"from": "autophagy_program",  "to": "MCH", "evidence_type": "germline",    "direction": "negative"},
    {"from": "G2M_phase_program",  "to": "MCH", "evidence_type": "germline",    "direction": "positive"},
    {"from": "MHC_class_II_program", "to": "RA",  "evidence_type": "viral",       "direction": "positive"},
    {"from": "MHC_class_II_program", "to": "SLE", "evidence_type": "viral",       "direction": "positive"},
    # CHIP → trait (CAD-specific)
    {"from": "DNMT3A_chip", "to": "CAD", "evidence_type": "somatic_chip", "direction": "positive"},
    {"from": "TET2_chip",   "to": "CAD", "evidence_type": "somatic_chip", "direction": "positive"},
    # Drug → program
    {"from": "statin_exposure", "to": "HMGCR_pathway", "evidence_type": "drug", "direction": "negative"},
    # IBD anchors — NOD2/IL23R germline LoF + anti-TNF drug RCT validation
    {"from": "NOD2",    "to": "IBD",           "evidence_type": "germline",    "direction": "positive"},
    {"from": "IL23R",   "to": "IBD",           "evidence_type": "germline",    "direction": "negative_lof"},
    {"from": "TNF",     "to": "IBD",           "evidence_type": "drug",        "direction": "positive"},
    {"from": "IL10",    "to": "IBD",           "evidence_type": "germline",    "direction": "negative_lof"},
    # AMD anchors — CFH (complement, p<1e-120 GWAS) + VEGFA (anti-VEGF drugs validate)
    {"from": "CFH",     "to": "AMD",           "evidence_type": "germline",    "direction": "positive"},
    {"from": "VEGFA",   "to": "AMD",           "evidence_type": "drug",        "direction": "positive"},
]


# ---------------------------------------------------------------------------
# Per-disease anchor subsets — single source of truth for recovery checks.
# causal_discovery_agent imports REQUIRED_ANCHORS_BY_DISEASE from here
# rather than maintaining its own hard-coded list.
# ---------------------------------------------------------------------------

# Maps disease full-name → short key used in REQUIRED_ANCHORS_BY_DISEASE.
# Extend this as new diseases are added.
_DISEASE_SHORT_NAMES_FOR_ANCHORS: dict[str, str] = {
    "coronary artery disease":      "CAD",
    "ischemic heart disease":       "CAD",
    "myocardial infarction":        "CAD",
    "heart disease":                "CAD",
    "rheumatoid arthritis":         "RA",
    "systemic lupus erythematosus": "SLE",
    "inflammatory bowel disease":   "IBD",
    "ibd":                          "IBD",
    "crohn's disease":              "IBD",
    "crohns disease":               "IBD",
    "ulcerative colitis":           "IBD",
    "age-related macular degeneration": "AMD",
    "macular degeneration":         "AMD",
    "amd":                          "AMD",
}

REQUIRED_ANCHORS_BY_DISEASE: dict[str, list[tuple[str, str]]] = {
    # Only genes with approved therapeutics for the indication.
    # Trait node must match the disease name written to the Kùzu graph,
    # not an intermediate phenotype (e.g. "LDL-C") — edges are gene→disease.
    "CAD": [
        ("PCSK9", "coronary artery disease"),   # evolocumab / alirocumab approved
        ("HMGCR", "coronary artery disease"),   # statins — approved first-line
    ],
    # AMD: VEGFA is the approved therapeutic target (ranibizumab, bevacizumab,
    # aflibercept — standard of care for wet AMD).
    "AMD": [
        ("VEGFA", "age-related macular degeneration"),
    ],
}
