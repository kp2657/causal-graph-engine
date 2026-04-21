# Data Sources

All data sources used by this pipeline are publicly accessible. No proprietary data is required. This document lists each source, what it provides, how to access it, and where it is used in the pipeline.

---

## GWAS summary statistics

### IEU OpenGWAS
- **URL:** https://gwas.mrcieu.ac.uk
- **Access:** Free JWT token (register via GitHub OAuth at https://api.opengwas.io)
- **Token expiry:** 14 days from issue; renew at the same URL
- **Key study IDs used:**
  | Study | Trait | N |
  |-------|-------|---|
  | `ieu-a-7` | Coronary artery disease (CARDIoGRAMplusC4D) | 547,261 |
  | `ieu-b-4816` | CAD (FinnGen R10) | 412,181 |
  | `ebi-a-GCST90014023` | Age-related macular degeneration | 456,438 |
  | `ebi-a-GCST90018926` | Inflammatory bowel disease | 590,186 |
  | `ieu-a-298` | LDL cholesterol | 94,595 |
- **Used for:** MR-IVW instruments, GWAS anchor gene validation
- **Config key:** `OPENGWAS_JWT` in `.env`

### GWAS Catalog
- **URL:** https://www.ebi.ac.uk/gwas/
- **Access:** Open REST API, no authentication
- **Used for:** Cross-reference of GWAS hits, study metadata

---

## Genetics / functional annotation

### gnomAD
- **URL:** https://gnomad.broadinstitute.org
- **API:** https://gnomad.broadinstitute.org/api (GraphQL, open)
- **Version:** gnomAD v4 (GRCh38)
- **Used for:** Population allele frequencies, constraint metrics (pLI, LOEUF), variant annotation

### GTEx
- **URL:** https://gtexportal.org
- **API:** https://gtexportal.org/rest/v1/ (open)
- **Version:** GTEx v8 (GRCh38)
- **Used for:** Tissue-specific eQTLs for eQTL–GWAS direction concordance (Tier 2.5)

### Open Targets Genetics (L2G)
- **URL:** https://genetics.opentargets.org
- **API:** GraphQL, open
- **Version:** Latest at query time
- **Used for:** GWAS locus-to-gene scores (L2G ≥ 0.10 threshold), variant fine-mapping

### Open Targets Platform
- **URL:** https://platform.opentargets.org
- **API:** https://api.platform.opentargets.org/api/v4/graphql (open)
- **Used for:** Target druggability, known drugs, max clinical phase, overall association scores

---

## Single-cell data

### CELLxGENE Census
- **URL:** https://cellxgene.cziscience.com/census
- **Access:** Open; Python API via `cellxgene-census` package
- **Census version:** `2025-11-08`
- **Datasets used:**
  | Disease | Cell type | Cells (disease / normal) |
  |---------|-----------|--------------------------|
  | AMD | RPE | 289 / 29,711 |
  | IBD | Colon epithelial | varies by run |
  | CAD | Coronary artery SMC | varies by run |
- **Used for:** Disease-state axis (log2FC), state_influence scores, GPS input signature
- **Cache location:** `data/cellxgene/{disease}/{disease}_{cell_type}.h5ad`
- **Note:** Files are ~1–10 GB per disease; not included in this repository

### Replogle 2022 Perturb-seq (RPE-1)
- **GEO accession:** GSE246756
- **URL:** https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE246756
- **Size:** ~50 GB (h5ad)
- **Used for:** β matrix (gene → program perturbation effects), Tier 1 interventional evidence
- **Local path:** `data/perturb_seq/` (must be downloaded separately; see README)
- **Citation:** Replogle JM et al. *Cell* 2022.

---

## Drug / chemistry

### ChEMBL
- **URL:** https://www.ebi.ac.uk/chembl/
- **API:** https://www.ebi.ac.uk/chembl/api/data/ (open REST)
- **Version:** Latest at query time (ChEMBL 34+)
- **Used for:** IC50/Ki values, compound bioactivity, target–drug associations

### PubChem
- **URL:** https://pubchem.ncbi.nlm.nih.gov
- **API:** https://pubchem.ncbi.nlm.nih.gov/rest/pug/ (open REST)
- **Used for:** SMILES retrieval, molecular formula, InChIKey, compound CID lookup

### Enamine REAL (GPS compound library)
- **URL:** https://enamine.net/compound-collections/real-compounds/real-database
- **Access:** Subset pre-computed in GPS Docker image
- **Used for:** GPS phenotypic screening compound library (Z-number identifiers)

---

## Clinical data

### ClinicalTrials.gov
- **URL:** https://clinicaltrials.gov
- **API:** https://clinicaltrials.gov/api/v2/ (open, no auth)
- **Used for:** Drug–disease trial phase, trial count per target, disease-specific validation
- **Note:** `countTotal=true` must be passed to receive `totalCount` in response

---

## Literature / knowledge bases

### NCBI E-utilities
- **URL:** https://eutils.ncbi.nlm.nih.gov
- **Access:** Free API key from https://www.ncbi.nlm.nih.gov/account/ (raises rate limit from 3 to 10 req/s)
- **Used for:** PubMed literature search, gene synonyms, OMIM
- **Config key:** `NCBI_API_KEY` in `.env`

### Semantic Scholar
- **URL:** https://www.semanticscholar.org/product/api
- **Access:** Free; optional API key for higher rate limits
- **Used for:** Paper search, citation graph, abstract retrieval
- **Config key:** `SEMANTIC_SCHOLAR_API_KEY` in `.env` (optional)

### Crossref
- **URL:** https://api.crossref.org
- **Access:** Open; "polite pool" via `mailto=` parameter (no key required)
- **Config key:** `CROSSREF_MAILTO` in `.env` (your email address)

---

## Pathway / knowledge graph

### MSigDB (Molecular Signatures Database)
- **URL:** https://www.gsea-msigdb.org/gsea/msigdb/
- **Version:** v2023.2 (Hallmarks H collection, C2 Reactome)
- **Access:** Free for academic use; GMT files downloadable
- **Used for:** NMF program annotation, pathway enrichment

### STRING / BioGRID (protein–protein interactions)
- **URL:** https://string-db.org / https://thebiogrid.org
- **Access:** Open (STRING API); free key for BioGRID
- **Used for:** KG completion — protein interaction edges

---

## Caching

All external API responses are cached in `data/api_cache.sqlite` (TTL: 7–30 days depending on source). The cache file is excluded from version control (in `.gitignore`). Delete it to force fresh API calls.

To clear a specific function's cache:

```python
import sqlite3, json, hashlib
# See pipelines/api_cache.py → _make_key() for key construction
conn = sqlite3.connect("data/api_cache.sqlite")
conn.execute("DELETE FROM cache_entries WHERE key = ?", (computed_key,))
conn.commit()
```
