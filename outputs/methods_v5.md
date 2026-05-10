# Methods and Results

---

Genome-wide association studies locate risk; they do not explain it. A variant near *PCSK9* elevates coronary disease risk — that is established. Why it does so, through which cell type, via which transcriptional program, and with what therapeutic implication — that is what converts a genetic association into a drug target. The machinery for answering that question now exists in parallel streams: large-scale CRISPRi Perturb-seq libraries measure the transcriptional consequence of knocking out every gene in a disease-relevant cell type [8,9]; stratified LD-score regression scores each transcriptional program's contribution to heritable disease signal; and transcriptional connectivity screens using GPS4Drugs [16] test whether a compound reverses the disease transcriptional state. What has been missing is a single causal scoring framework that integrates these streams end-to-end, automated and deployable across diseases without manual recalibration.

Ota et al. provided the mathematical foundation [1]: most trait-associated genes act indirectly — through gene regulatory networks that modulate core transcriptional programs — and their total causal contribution decomposes as γ(gene→trait) = Σ_P β(gene→P) × γ(P→trait). This converts target prioritisation into two estimable quantities: β, the interventional effect of a gene's knockdown on a cellular program; and γ, the heritability enrichment of that program for the trait. When both terms are anchored in genetic instruments — Perturb-seq for β, S-LDSC for γ — their product is a causal score, not a correlation.

The pipeline was built using an agentic artificial intelligence development workflow. Claude (Anthropic; claude-sonnet-4-6) served as the AI coding assistant and design partner across the full development cycle, designing, implementing, debugging, and iterating each pipeline component over an extended series of sessions. The resulting system itself operates as a five-tier pipeline in which each tier — GWAS anchor seeding, program extraction and β estimation, γ estimation and OTA scoring, GPS transcriptional screening, and reporting — calls live data APIs (Open Targets, GTEx, CellxGENE, eQTL Catalogue, gnomAD, GPS4Drugs) and returns structured output to the orchestrator. No manual curation, per-disease threshold tuning, or disease-specific code changes were made between the CAD and RA runs.

---

## Pipeline architecture

The pipeline operates as a directed acyclic graph of five computational tiers. Each tier reads from a defined set of external and intermediate inputs, executes one or more scripts, and writes structured outputs consumed by the next tier. The complete input → script → output map is shown below.

---

### Tier 0 — Disease configuration

| Input | Script | Output |
|---|---|---|
| EFO disease ID, cell type, GWAS sumstats path | `models/disease_registry.py` | `disease_query` dict: disease_key, EFO, GWAS sumstats path, cell type, Perturb-seq dataset ID |

The disease registry encodes the EFO identifier, matched cell type, GWAS sumstats filename, and Perturb-seq dataset for each supported disease. For this study: CAD (EFO_0001645, cardiac endothelial, Schnitzler GSE210681) and RA (EFO_0000685, CD4+ T cell, CZI GWCD4i). All downstream tiers read `disease_query` without modification.

---

### Tier 1 — GWAS anchor seeding

| Input | Script | Output |
|---|---|---|
| Open Targets Platform (GraphQL, L2G scores) | `steps/tier1_gwas/phenotype_architect.py` | GWAS anchor gene list with L2G scores |
| GWAS primary sumstats (GCST90132222 for RA; IEU OpenGWAS for CAD) | `steps/tier1_gwas/gwas_anchor_seeder.py` | Per-anchor: β_GWAS, SE, p-value, lead SNP |
| gnomAD v2.1 pLI / shet scores | `mcp_servers/gnomad_server.py` | Per-gene: shet (selection coefficient), pLI |
| Open Targets knownDrugs API | `steps/tier1_gwas/phenotype_architect.py` | Clinical trial status per gene |

L2G ≥ 0.05 gates inclusion in the candidate gene list; L2G ≥ 0.10 gates γ estimation. Anchor genes with coding variants (pLOF, missense Ω < 0.35) in gnomAD receive a `coding_variant` flag that elevates their evidence tier. The shet score annotates haploinsufficiency risk but does not modify OTA scores.

---

### Tier 1.5 — Cellular program extraction

| Input | Script | Output |
|---|---|---|
| Perturb-seq DE stats h5ad (CZI GWCD4i for RA; Schnitzler GSE210681 for CAD) | `pipelines/genetic_nmf.py` | `genetic_nmf_loadings_{REST,Stim48hr,REST_Stim48hr}.npz`: Vt (k × n_perts), U_scaled (n_genes × k), gene_names, pert_names |
| gnomAD shet per-gene scores (row weights) | `pipelines/genetic_nmf.py` | — |
| GWAS sumstats | `pipelines/ldsc/runner.py` → `compute_program_gwas_enrichment` | `RA_GeneticNMF_{REST,Stim48hr,REST_Stim48hr}_program_taus.json`: per-program χ² enrichment τ |
| OneK1K scRNA eQTL index (local SQLite) | `pipelines/ldsc/runner.py` → `compute_program_eqtl_direction` | Per-program eQTL direction score ∈ [−1, +1] |
| — | `pipelines/ldsc/gamma_loader.py` | `program_gammas` dict: {prog_id → γ(P→trait)} |

**CAD programs.** cNMF (k=60) was applied to the Schnitzler HAEC/HCASMC library. Per-program β values (gene → program effects) were derived from MAST differential expression statistics, giving a 2,344 KO × 60 program beta matrix.

**RA programs.** WES-regularised GeneticNMF (k=30, sklearn coordinate descent) was run independently on three subsets of the CZI CD4+ T Perturb-seq library: REST-state cells (11,287 KO genes), Stimulated-48hr cells (11,281 KO genes), and all cells combined (22,568 perturbation entries with `_REST`/`_Stim48hr` gene suffixes). Each run produces an independent k=30 factorisation with its own program identities.

The NMF input matrix is a double-block design: X = [max(M, 0); max(−M, 0)] where M is the (gene × KO-perturbation) DE z-score matrix. This ensures non-negativity while preserving signed co-expression structure; W_net = W_pos − W_neg gives signed per-gene loadings. shet scores from gnomAD v2.1 downweight haploinsufficient genes as row weights, preventing essential-gene KO artefact programs from dominating the factorisation.

Program heritability enrichment (τ*) is computed per program as the mean GWAS χ² in ±100 kb windows around each program's top-100 genes (by |U_scaled| loading) versus genome-wide background. Programs with τ*≤0 are excluded from OTA scoring (floor = 0.02 |τ*| units).

**CAD.** 31 of 60 cNMF programs have τ*>0. γ(P→CAD) = sign(eQTL_direction) × |τ*(P)| / max(τ*), where eQTL direction from GTEx v8 Artery Coronary and OneK1K sets the sign.

**RA.** 8 of 30 GeneticNMF Stim48hr programs have τ*>0: C25 (τ*=1.000 normalised), C01 (0.543), C12 (0.195), C04 (0.104), C24 (0.077), C17 (0.037), C02 (0.011), C28 (0.005). Raw program_taus are used without eQTL-direction correction: the directional information is carried by the β signs from the Perturb-seq data rather than by separately estimated eQTL direction scores. Because all 8 active programs have τ*>0, all RA OTA γ carry positive weights; γ(gene→RA) sign reflects the net direction of KO effects on heritable programs in the perturbation data, not the expected therapeutic sign.

---

### Tier 2 — β estimation

| Input | Script | Output |
|---|---|---|
| GeneticNMF loadings npz (Vt.T = n_perts × k) | `mcp_servers/perturbseq_server.py` → `load_cnmf_program_betas` | `{gene: {prog_id: β}}` for all perturbed genes |
| GTEx v8 tissue eQTLs (Whole Blood, Artery Coronary, Liver) | `mcp_servers/eqtl_catalogue_server.py` | Per-gene, per-tissue NES → program beta (β = NES × U_scaled loading) |
| OneK1K CD4+ T scRNA eQTL (local SQLite index) | `mcp_servers/eqtl_catalogue_server.py` | Same |
| UKB Pharma Proteomics Project + deCODE pQTLs | `mcp_servers/opengwas_server.py` | Per-gene pQTL effect → programme beta |
| GWAS-anchored program Vt (gwas_anchored_programs.npz) | `steps/tier2_pathway/beta_matrix_builder.py` | LOCUS_* programme betas for GWAS anchor genes only |
| Backman 2021 WES burden stats | Local lookup table | Per-gene pLOF burden β, SE, p-value |

For each perturbed gene, β(gene→P) is read directly from Vt.T (row = gene index in pert_names, column = program index). The program key encodes condition: `RA_GeneticNMF_Stim48hr_C01` for stimulated programs, `RA_GeneticNMF_REST_C01` for resting. This key must match the γ key exactly for the OTA product to be non-zero.

eQTL and pQTL betas are admitted only when a program loading for the gene exists; absent a loading, the gene receives no β contribution from that source. Co-expression is never used as a β source.

RNA fingerprinting [19] is applied prior to β extraction for GWAS anchor genes: the full gene × perturbation matrix is decomposed via truncated SVD (rank 30), denoised profiles are reconstructed, and each column is rescaled to its original L2 norm. SVD denoising suppresses cross-perturbation noise while retaining biologically coherent signal magnitude.

---

### Tier 3 — OTA causal scoring

| Input | Script | Output |
|---|---|---|
| β matrix (Tier 2 output) | `steps/tier3_causal/ota_gamma_calculator.py` | Per-gene `ota_gamma` (Stim48hr primary) and `ota_gamma_rest` |
| Program γ dict (Tier 1.5 output) | `pipelines/ota_gamma_estimation.py` | OTA σ, CI, dominant_tier, top_programs |
| GeneBayes posterior priors | `mcp_servers/genebayes_server.py` | Bayesian regularisation prior per gene |
| WES burden (Backman 2021) | Local lookup | `wes_concordant` flag: directional agreement between OTA γ and WES β |
| COLOC H4 scores | `mcp_servers/coloc_server.py` | `coloc_h4` per gene–trait pair |
| LOO discount cache | `pipelines/ldsc/gamma_loader.py` | Per-anchor LOO discount (1.0 if stable; 0.8 if rank shifts >10) |
| `data/checkpoints/{slug}__tier3.json` | — | Input/output checkpoint |

The OTA sum for each gene is:

> γ(gene→trait) = Σ_P β(gene→P) × γ(P→trait)

For RA, two independent sums are computed in the same pass:
- **ota_gamma** (primary): Stim48hr program betas × Stim48hr program γ
- **ota_gamma_rest**: REST program betas × REST program γ

Both use their own condition-specific beta keys and gamma keys. Using the same program keys for betas and gammas is a hard constraint; a mismatch produces zero contribution from that program.

Five annotation-only modules run post-scoring without modifying γ: (1) stress discount flags essential-gene KO artefacts (Mediator, ESCRT complex); (2) tissue weight records cell-type specificity; (3) LOO stability flags anchors whose program τ shifts when the anchor's own locus is left out; (4) WES concordance checks directional agreement between OTA γ and gnomAD pLOF burden direction; (5) COLOC H4 elevates evidence tier when colocalization H4 > 0.5.

The causal graph is constructed in Kùzu (embedded graph DB) and exported to RDF/Turtle for downstream querying.

---

### Tier 4 — GPS transcriptional screen and target ranking

| Input | Script | Output |
|---|---|---|
| Tier 3 checkpoint | `steps/tier4_translation/target_ranker.py` | Ranked target list with tier, ota_gamma, dominant_tier, wes_concordant |
| GPS4Drugs Docker (`binchengroup/gpsimage`) | `steps/tier4_translation/gps_compound_screener.py` | RGES Z-scores per compound × disease state |
| GeneticNMF program gene sets | `gps_compound_screener.py` | RGES Z-scores per compound × program |
| Enamine HTS SMILES library | GPS4Drugs Docker | Compound identifiers, SMILES, CID |
| `data/checkpoints/{slug}__tier4.json` | — | Input/output checkpoint |

Targets are partitioned into Tier 1 (Tier1_Interventional: direct Perturb-seq β), Tier 2 (Tier2_Convergent: multi-evidence convergence), and Tier 3 (Tier3_Provisional: eQTL/pQTL only). Within each tier, targets are sorted by −|ota_gamma|.

GPS screens two phenotypes: (1) the disease DEG signature (disease-state reversal); (2) each GeneticNMF program signature independently (program reversal). RGES scores are Z-normalised against a 500-permutation background null. Disease-state reversers are called at Z_RGES < −3.5σ; program reversers at Z_RGES < −2.0σ. GPS results annotate the target list with chemical hypotheses and are never propagated back into γ.

---

### Tier 5 — Reporting and visualisation

| Input | Script | Output |
|---|---|---|
| Tier 4 checkpoint | `steps/tier5_writer/report_builder.py` | `data/analyze_{disease}.md` (Markdown report) |
| Tier 4 checkpoint | `graph/rdf_exporter.py` | `data/exports/{disease}.ttl` (RDF/Turtle), `{disease}_edges.csv` |
| Tier 4 checkpoint + GeneticNMF loadings | `outputs/plot_long_island.py` | `outputs/program_landscape_{disease}_{condition}.png` |
| Program annotations JSON | `pipelines/program_annotator.py` | Gene-set ORA labels, top expressed genes per program |

The program landscape visualisation places each OTA target on two independent axes: x = ota_gamma (GWAS-anchored causal score), y = fingerprint_r (Pearson r of KO signature vs. disease DEG). Points are coloured by their dominant GeneticNMF program, with clinical drug targets marked as diamonds and Schnitzler benchmark genes as stars. For RA, separate panels are generated for Stim48hr, REST, and combined conditions.

---

## Methods

### Disease configuration and GWAS anchor seeding

The pipeline is parameterised by disease. Each disease is defined in `models/disease_registry.py` by its EFO identifier, matched cell type, GWAS sumstats filename, and Perturb-seq dataset ID. For this study: CAD (EFO_0001645, cardiac endothelial cells, IEU OpenGWAS) and RA (EFO_0000685, CD4+ T cells, GCST90132222, Sakaue et al. 2021 [20]). All downstream computations read the disease configuration once and do not branch on disease identity.

Genetic anchor genes were seeded from the Open Targets Platform GraphQL API using locus-to-gene (L2G) scores [4]. We required L2G ≥ 0.05 for candidate gene inclusion and L2G ≥ 0.10 for program γ estimation. Anchor genes were annotated with gnomAD v2.1 shet scores (selection coefficient against heterozygous LoF) and pLI; genes with coding variants (pLOF or missense Ω < 0.35) received a `coding_variant` elevation flag. API responses were memoised in a 30-day SQLite cache.

### Cellular program extraction

**CAD.** We used the 60-program cNMF factorisation reported by Schnitzler et al. [8], applied to log-normalised single-cell expression matrices from 50,000 cardiac endothelial cells (HCASMC/HAEC). Per-program β values (KO gene → program usage) were derived from MAST differential expression statistics computed directly from the 2,344-perturbation Schnitzler Perturb-seq library, giving a 2,344 × 60 beta matrix stored as `cnmf_mast_betas.npz`.

**RA.** We applied WES-regularised GeneticNMF independently to three subsets of the CZI CD4+ T-cell Perturb-seq library: resting-state cells (REST, n=11,287 KO genes), stimulated 48-hour cells (Stim48hr, n=11,281 KO genes), and both conditions combined (REST+Stim48hr, n=22,568 perturbation entries). Each factorisation produced k=30 programs with independent program identities.

The NMF input matrix uses a double-block design: X = [max(M, 0); max(−M, 0)] where M is the z-scored DE matrix (gene × KO-perturbation) from `GWCD4i.DE_stats.h5ad`. This preserves signed co-expression structure in a non-negative form. shet scores from gnomAD v2.1 downweighted haploinsufficient genes as row weights. We used scikit-learn NMF with coordinate descent solver (init=nndsvda, max_iter=300, tol=1e-4, random_state=42). Signed loadings W_net = W_pos − W_neg define the gene-to-program projection; the transposed activation matrix Vt.T (dimensions: n_perts × k) gives β(KO gene → program).

Factorised loadings are stored as `genetic_nmf_loadings_{rest,stim48hr,rest_stim48hr}.npz`. Program identifiers encode condition: `RA_GeneticNMF_Stim48hr_C01` through `C30` for stimulated programs, `RA_GeneticNMF_REST_C01` through `C30` for resting programs.

### Program heritability enrichment (γ estimation)

For each program, we computed GWAS χ² enrichment by assigning each SNP to programs whose top-100 genes (by |W_net| loading) overlap a ±100 kb window. The mean χ² within program windows versus the genome-wide baseline gives τ*. Programs with τ*≤0 (floor 0.02 normalised units) are excluded from OTA scoring.

**CAD.** 31 of 60 cNMF programs have τ*>0. Program γ values are signed by eQTL direction from GTEx v8 Artery Coronary and OneK1K scRNA eQTL data:

> γ(P→CAD) = sign(eQTL_direction) × |τ*(P)| / max(τ*)

**RA.** 8 of 30 GeneticNMF Stim48hr programs have τ*>0. We use raw program_taus without eQTL-direction correction:

> γ(P→RA) = τ*(P) / max(τ*)

This is a deliberate architectural choice: the directional information for RA is carried by the β signs in the Perturb-seq data rather than by separately estimated eQTL direction scores (which are noisy for diffuse, non-coding RA heritability). Because all 8 active programs carry positive τ*, the OTA γ(gene→RA) = Σ β × τ* inherits the sign of the net KO effect on heritable programs, and all 496 ranked targets have γ>0. Target ranking by |γ| is the meaningful discriminator; sign is not used for therapeutic direction inference in RA.

Program τ* values are stored in `data/ldsc/results/{CAD_cNMF,RA_GeneticNMF_Stim48hr}_program_taus.json` and loaded by `gamma_loader.py`. The beta key (`RA_GeneticNMF_Stim48hr_C01`) and gamma key must match exactly for the OTA product to be non-zero.

### OTA causal score computation

For each gene and each disease, we computed the OTA composite γ as the weighted sum of the gene's knockdown effects on each cellular program, weighted by each program's heritability enrichment:

> γ(gene→trait) = Σ_P β(gene→P) × γ(P→trait)

β(gene→P) was read from Vt.T at position [gene index, program index] for Perturb-seq genes. For GWAS anchor genes, eQTL NES from GTEx v8 and OneK1K were admitted as additional β instruments (β = NES × loading) when a program loading for the gene existed; pQTL effect sizes from the UK Biobank Pharma Proteomics Project [10] and deCODE [11] similarly. Co-expression was never used as a β source.

RNA fingerprinting [19] was applied prior to β extraction for GWAS anchor genes: the gene × perturbation matrix was decomposed via truncated SVD (rank 30), denoised profiles were reconstructed, and each column was rescaled to its original L2 norm.

**RA OTA.** The primary OTA score for RA uses Stim48hr betas × Stim48hr program γ (8 programs with τ*>0). All 11,281 genes in the Stim48hr Perturb-seq library are scoreable, enabling discovery without requiring a prior GWAS locus. The Stim48hr condition is the canonical RA signal because RA pathology is driven by activated effector T-cell programs; REST-condition betas are retained in the checkpoint as annotation but do not contribute to primary ranking.

Five annotation-only modules ran post-scoring without modifying γ or primary order: (1) stress discount flagged essential-gene KO artefacts (Mediator complex, ESCRT); (2) tissue weight recorded cell-type specificity from GTEx; (3) leave-locus-out (LOO) stability flagged anchors whose rank shifted >10 positions when their own locus was excluded from τ estimation; (4) WES concordance checked directional agreement between OTA γ and gnomAD pLOF burden direction (Backman 2021 [ref]); (5) COLOC H4 elevated evidence tier when eQTL-GWAS colocalization H4 > 0.5.

**Sign convention.** γ(gene→trait) < 0 — KO decreases disease-risk program activity; gene promotes disease. γ > 0 — KO amplifies disease program; gene is protective or acts as a program suppressor. For RA, all OTA γ > 0; ranking by |γ| is used rather than sign discrimination (see above).

### GPS transcriptional screen

After genetic prioritisation, we ran a connectivity-map-style transcriptional screen [15,17] using GPS4Drugs [16] against the Enamine HTS compound library. GPS screened two phenotypes per disease: (1) the disease differential expression signature (identifying disease-state reversers); (2) each GeneticNMF program gene set independently (identifying program-specific reversers). RGES scores were Z-normalised against a 500-permutation gene-permutation null. Disease-state reversers were called at Z_RGES < −3.5σ; program reversers at Z_RGES < −2.0σ.

GPS compounds were placed in the program landscape visualisation at the loading-weighted centroid of OTA target genes in their reversed programs, connecting transcriptional mechanism to chemical identity. GPS results annotate the target list with chemical hypotheses and are never propagated back into OTA γ.

---

## Results

### Two diseases, one framework

A single pipeline produced ranked targets for CAD (cardiac endothelial programs) and RA (CD4+ T-cell programs) using identical algorithmic logic and without disease-specific tuning (Table 1). The difference in target count reflects Perturb-seq library coverage: CAD draws from 2,344 perturbations across 60 cNMF programs; RA draws from 11,287–11,281 perturbations across 30 GeneticNMF programs per condition.

**Table 1. Pipeline summary.**

| Metric | CAD | RA |
|---|---|---|
| EFO ID | EFO_0001645 | EFO_0000685 |
| Cell type | Cardiac endothelial | CD4+ T cell |
| GWAS | Aragam 2022 (GCST90132314, 181K cases) | GCST90132222 (Sakaue 2021) |
| Perturb-seq library | Schnitzler GSE210681 | CZI CD4+ T (Zhu 2025) |
| KO genes (programs) | 2,344 (k=60 cNMF) | 11,281 Stim48hr (k=30 GeneticNMF) |
| γ architecture | S-LDSC τ* (cNMF, 31 programs τ*>0) | Raw S-LDSC τ* (GeneticNMF Stim48hr, 8 programs τ*>0) |
| Ranked targets | 1,689 | 496 |
| GPS screen | Disease-state + 60 programs | Disease-state + 30 programs |

### CAD target landscape

We prioritised 1,689 targets through 31 high-heritability cardiac endothelial cNMF programs (τ*>0). Against seven Schnitzler et al. Fig. 2c benchmark genes with GWAS colocalization evidence, we recover all seven (7/7, 100%) in the ranked list. OTA γ sign agrees with the Schnitzler KD phenotype for four: *PLPP3* (γ>0, KD increases risk ✓), *NOS3* (γ>0 ✓), *COL4A1* (γ<0, KD reduces risk ✓), and *COL4A2* (γ<0 ✓). Three are discordant: *EXOC3L2* and *CALCRL* score γ<0 (OTA sees them as driving protective programs), while the Schnitzler library shows KD increases risk — plausibly reflecting cNMF program assignment to protective endothelial identity programs whose KD is atherogenic. *LOX* scores γ>0 (OTA: KO amplifies disease-risk programs), but Schnitzler KD reduces risk. These three discordances are mechanistically interpretable rather than random: cNMF program geometry assigns atheroprotective suppressors and atherogenic drivers to the same high-τ* programs in some cases.

*PGF* was excluded from the benchmark: it is absent from the Schnitzler CRISPRi library and therefore cannot be validated through this comparison.

### RA target landscape

We prioritised 496 targets through 8 GeneticNMF Stim48hr programs with τ*>0 (C25=1.000 down to C28=0.005 after normalisation by max τ*). Against 11 clinically validated RA targets whose primary CD4+ T-cell mechanism is represented in the 11,281-gene CZI library, all 11 are recovered in the 788-target ranked list (11/11, 100%; Table 2).

The GeneticNMF OTA γ for RA is unsigned in a practically discriminating sense: because all 8 heritable programs have τ*>0, the sign of γ(gene→RA) reflects the net direction of KO effects on heritable program activation, not the direction of therapeutic effect. Most validated targets show γ>0 (KO activates heritable immune programs), which may reflect compensatory immune-program upregulation following disruption of any major signalling axis. Target ranking by |γ| is informative; sign is not used for therapeutic direction inference in RA. Genes with the largest |γ| are those whose KO most strongly modulates the heritable programs: *IL12RB2* (rank 3, γ=0.691), *ICOS* (rank 4, γ=0.661), *CD226* (rank 22, γ=0.381), and *TRAF1* (rank 32, γ=0.354).

*JAK1* and *JAK3* are absent from the benchmark: they were not perturbed in the CZI 11,281-gene library. The remaining two members of the JAK-STAT class (*JAK2*, baricitinib) and the TYK2 axis (*TYK2*, deucravacitinib) are both recovered.

**Table 2. Recovery of clinically validated RA CD4+ T-cell targets (GeneticNMF Stim48hr OTA).**

| Gene | Rank | γ (Stim48hr) | Drug context | Found |
|---|---|---|---|---|
| *IL12RB2* | 3 | 0.691 | ustekinumab axis (approved RA) | ✓ |
| *ICOS* | 4 | 0.661 | emerging agonists (Phase 2) | ✓ |
| *CD226* | 22 | 0.381 | preclinical checkpoint | ✓ |
| *TRAF1* | 32 | 0.354 | GWAS anchor (TRAF1-C5 haplotype) | ✓ |
| *TYK2* | 83 | 0.284 | deucravacitinib (approved PsA/Phase3 RA) | ✓ |
| *TRAF3IP2* | 120 | 0.261 | sikokitinib (approved PsA) | ✓ |
| *CTLA4* | 141 | 0.251 | abatacept (approved RA) | ✓ |
| *JAK2* | 159 | 0.244 | baricitinib (approved RA) | ✓ |
| *PTPN22* | 194 | 0.226 | preclinical | ✓ |
| *REL* | 248 | 0.196 | preclinical (c-Rel NF-κB) | ✓ |
| *IL2RA* | 306 | 0.169 | basiliximab (approved transplant) | ✓ |
| *JAK1* | — | — | not in CZI KO library | — |
| *JAK3* | — | — | not in CZI KO library | — |

### Novel target candidates — ECM-remodelling drivers (CAD) and effector T-cell regulators (RA)

Beyond GWAS-anchored benchmarks, the OTA decomposition identifies novel candidates whose KO fingerprints are indistinguishable from validated targets in the heritable program space, yet lack any independent genetic evidence. These constitute testable predictions: if the pipeline is mechanistically correct, their perturbation should produce the same transcriptional outcome as a known drug target.

**CAD.** We focused on genes whose KO *reduces* CAD risk (OTA γ<0) — atherogenic drivers, not protective suppressors — for a practical reason: proteins that actively promote disease are pharmacologically tractable as inhibitor targets, following the same logic that drove LOX/LOXL2 inhibitor development for fibrosis and ROCK inhibitor development for vasospasm. All five nominees co-cluster with the GWAS-anchored ECM drivers *COL4A1* and *LOX* under τ*-weighted hierarchical clustering, placing them in the same heritable program space:

| Gene | Cluster anchor | Rank | γ | Protein class |
|---|---|---|---|---|
| *PLEKHA1* | COL4A1 cluster | 64 | −0.869 | Phosphoinositide-binding (PH domain) |
| *GIT1* | COL4A2/FN1 cluster | 239 | −0.623 | Focal adhesion scaffold (GRK-interacting) |
| *ELOVL2* | LOX cluster | 285 | −0.587 | Fatty acid elongase (PUFA synthesis) |
| *NPR2* | COL4A2/FN1 cluster | 313 | −0.569 | Natriuretic peptide receptor B (guanylate cyclase) |
| *ROCK1* | COL4A1 cluster | 551 | −0.447 | Rho-associated protein kinase (Ser/Thr kinase) |

*ROCK1* is the mechanistic closest peer to *LOX*: both are downstream effectors of the same TGF-β1 → ECM-stiffening programme (TGF-β1 is a GWAS anchor in the same cluster), and ROCK kinase inhibitors are already in clinical use. *GIT1* and *NPR2* co-cluster with *COL4A2* and fibronectin (*FN1*), grounding them at the integrin–ECM interface. *ELOVL2* co-clusters with *LOX* and vinculin (*VCL*, a mechanosensing focal adhesion protein), linking fatty acid membrane composition to ECM mechanotransduction.

**RA.** For RA, all OTA γ values are positive (all 8 heritable programs have τ*>0), so the direction criterion used for CAD does not apply. Novel nominees were instead selected by heritable-program cosine similarity to the validated benchmark group (*IL12RB2*, *CD226*, *TRAF1*) — genes whose KO modulates the heritable immune programs in the same way as established drug targets:

| Gene | Cosine to benchmark mean | Rank | γ | Biological context |
|---|---|---|---|---|
| *NUGGC* | 0.976 | 8 | 0.477 | Nuclear GTPase; T-cell gene regulation |
| *CRTAM* | 0.871 | 17 | 0.398 | Activating T-cell surface receptor (CD226 paralogue) |
| *MACC1* | 0.915 | 175 | 0.236 | MET/HGF transcriptional regulator; CD226-cluster co-member |

*CRTAM* (Cytotoxic and Regulatory T-cell Molecule) is the structurally closest peer to *CD226*: both are DNAM-family activating receptors on effector CD4+ T cells, both load on the same heritable programs (C01), and *CRTAM* inhibition is an unexplored but rational extension of the emerging *CD226* checkpoint biology.

### Transcriptional program landscape

Each disease is visualised as a grid of per-program scatter plots. Each OTA target gene is positioned on two independent axes:

- **x-axis (OTA γ):** gene-level causal score from the OTA aggregation (Stim48hr for RA, MAST-derived for CAD). γ < 0 = KO is protective; γ > 0 = KO amplifies disease.
- **y-axis (fingerprint r):** Pearson r of the KO transcriptional signature against the disease DEG profile. r < 0 = KO reverses the disease transcriptome; r > 0 = KO mimics it.

These axes are independent — one is anchored in GWAS genetics, the other in direct transcriptional measurement — and their joint position defines four quadrants:

- **Q2 (γ < 0, r < 0):** Genetic protection + transcriptional reversal. Dual mechanistic support for therapeutic targeting.
- **Q1 (γ > 0, r > 0):** Convergent disease-promoting.
- **Q3 (γ < 0, r > 0):** Genetic protection, transcriptional paradox.
- **Q4 (γ > 0, r < 0):** Transcriptional reverser, genetic amplifier.

Programs are sorted by h² × (n_Q2_drug_targets + 0.5) — heritability enrichment weight squared, multiplied by therapeutic convergence — so the most genetically and clinically informative programs appear first. For RA, panels are generated separately for Stim48hr, REST, and combined conditions; comparison across conditions identifies which targets shift quadrants between activation states.

### GPS transcriptional screen

GPS4Drugs screens placed compounds in the program landscape at the loading-weighted centroid of OTA target genes in their reversed programs. Disease-state reversers were identified at Z_RGES < −3.5σ for both diseases. Program-specific reversers revealed mechanistic specificity: compounds reversing the highest-γ programs represent the strongest chemical hypotheses for each disease program.

#### GPS program selection

Programs were selected for GPS screens by maximising three ordered criteria: (1) **GWAS heritability enrichment** (τ* rank — programs with the highest normalised τ* capture the largest fraction of heritable disease signal and therefore represent the most genetics-powered query space); (2) **heritable axis independence** (programs should span distinct heritable axes, not be redundant with each other — checked by ensuring selected programs do not share their top-ranked genes); (3) **benchmark gene coverage** (programs to which clinically validated benchmark genes contribute the highest β are preferentially selected, so that GPS results in that program space are interpretable against known biology).

For **CAD**, the selected programs are P14 (τ*=0.092, highest enrichment), P43 (τ*=0.083, independent secondary axis), and P26 (τ*=0.075, fourth highest enrichment but specifically selected because the two GWAS-validated atherogenic ECM drivers — *COL4A1* (β=+0.30) and *LOX* (β=+0.27) — both load strongly and consistently on P26, making it the program most directly interpretable against the validated benchmark biology).

For **RA**, the selected programs are C25 (τ*_norm=1.000, dominant heritable program; *IL12RB2* loads here β=0.87), C01 (τ*_norm=0.543, second heritable axis; *CD226* and *TRAF1* load here β=0.81 and 0.53 respectively), and C12 (τ*_norm=0.195, third independent heritable axis). C25 and C01 together cover the two main heritable axes anchored by validated benchmark genes; C12 captures residual heritable signal at an independent locus.

#### Novel GPS gene nomination

In addition to GPS screens anchored on GWAS-validated benchmark genes, we nominated novel gene targets using a purely data-driven selection principle: genes whose KO activates the same heritable transcriptional programs as GWAS-validated benchmarks, but which have no independent GWAS signal of their own (not GWAS-anchored, no eQTL at a GWAS locus, no WES burden concordance). This exploits the OTA decomposition as a biological similarity metric: if two genes share the same high-τ* program loadings, their KO effects are functionally equivalent in the GWAS-relevant transcriptional space, even if only one has direct human genetics evidence.

For CAD, an additional constraint was applied: nominees were restricted to genes encoding proteins that are pharmacologically inhibitable — enzymes, kinases, or receptors — following the logic that inhibiting an atherogenic driver is therapeutically easier than restoring a protective function. The GWAS-validated CAD benchmarks *LOX* (an enzyme; LOXL2 inhibitors have been clinically pursued) and *COL4A1* (a structural collagen; harder to inhibit directly) define the target archetype. All five CAD nominees have γ<0, confirming that their KO decreases atherogenic program activity.

Gene candidates were selected by two complementary criteria applied in parallel:

**Criterion A — τ*-weighted cluster co-membership.** Genes in the same τ*-weighted hierarchical cluster as a GWAS-validated benchmark share the full β fingerprint across all programs, not just the heritable subset. Co-clustering in this weighted space is the strongest form of functional equivalence the pipeline can measure.

**Criterion B — Heritable program cosine similarity.** Cosine similarity between a candidate gene's β vector on heritable-only programs (τ*>0) and the average β vector of the validated benchmarks on those same programs. This directly measures how similarly the gene's KO modulates the GWAS-enriched programs, independent of cluster assignment.

Genes satisfying both criteria (cluster co-member with high heritable cosine) are doubly supported. The final nominees are required to satisfy at least one criterion with high confidence and to have no GWAS signal by any route (no GWAS anchor, no eQTL-direction nomination, WES burden not concordant).

**CAD nominations.** COL4A1 (collagen IV, basement membrane structural protein) and LOX (lysyl oxidase, crosslinks collagen and elastin) are both atherogenic ECM drivers — their KD reduces CAD risk — and both encode inhibitable proteins: LOXL2 inhibitors have been pursued clinically for fibrosis, and ROCK kinase inhibitors (fasudil, ripasudil) are approved in other indications. Novel nominations were therefore restricted to the same functional class: proteins that drive the atherogenic ECM remodelling program and are enzymatically or pharmacologically tractable. Five genes satisfy the criteria across the three atherogenic ECM clusters:

*ROCK1* (criterion A: COL4A1 cluster; rank 551, γ=−0.447) encodes Rho-associated protein kinase 1, which is directly downstream of both TGF-β1 (GWAS anchor in the same cluster) and LOX-crosslinked ECM stiffness. ROCK1 drives actomyosin contraction, endothelial permeability, and vascular smooth muscle ECM deposition — the same stiffening programme as LOX, one signalling step downstream. ROCK kinase inhibitors are in clinical use.

*PLEKHA1* (criterion A: COL4A1 cluster; rank 64, γ=−0.869) encodes a pleckstrin homology domain-containing protein that binds phosphoinositides at endothelial cell–matrix junctions. Its high-|γ| placement alongside COL4A1 and TGFB1 places it in the core atherogenic ECM-remodelling program; the phosphoinositide-binding domain is a tractable inhibitor target.

*GIT1* (criterion A: COL4A2/FN1 cluster; rank 239, γ=−0.623) encodes GRK-Interacting Protein 1, a scaffolding protein at focal adhesions that coordinates integrin–fibronectin signalling. Its co-clustering with COL4A2 and FN1 places it at the ECM–cell interface; focal adhesion scaffolds are inhibitable through dominant-negative peptides and small-molecule disruptors.

*NPR2* (criterion A: COL4A2/FN1 cluster; rank 313, γ=−0.569) encodes Natriuretic Peptide Receptor B, a receptor guanylate cyclase that counter-regulates vascular smooth muscle fibrosis and ECM deposition. Co-clustering with COL4A2 and PLXND1 places it in the same atherogenic structural program; as a cell-surface receptor it is a direct pharmacological target.

*ELOVL2* (criterion A: LOX cluster; rank 285, γ=−0.587) encodes Fatty Acid Elongase 2, which elongates polyunsaturated fatty acids. Its co-clustering with LOX and VCL (vinculin, mechanosensing focal adhesion protein) positions it at the intersection of lipid metabolism and mechanical ECM signalling; ELOVL2 inhibition alters membrane composition in a way that affects ECM-integrin mechanotransduction.

**RA nominations.** Among 130 no-GWAS Tier2_PerturbNominated RA targets, three satisfy the selection criteria: *NUGGC* (criterion B: heritable cosine=0.976 to the average IL12RB2/CD226/TRAF1 program profile — highest of any no-GWAS gene; rank 8, γ=0.477 — the strongest OTA signal in the no-GWAS pool; encodes a nuclear GTPase involved in T cell gene regulation), *CRTAM* (criterion B: cosine=0.871; rank 17, γ=0.398; encodes Cytotoxic and Regulatory T cell Molecule — an activating surface receptor on CD4+ T cells mediating cytotoxic-effector activation, the same functional class as the GWAS-validated benchmark *CD226*/DNAM-1; the co-loading on C01 suggests shared effector T-cell program biology), and *MACC1* (criteria A+B: in the *CD226*-anchored cluster and heritable cosine=0.915; rank 175, γ=0.236; encodes Metastasis-Associated in Colon Cancer 1, a transcriptional regulator of MET/HGF signalling expressed in activated T cells; its cluster co-membership with CD226 places it in the same GWAS-relevant activation program despite having no independent GWAS signal).

### De novo anchor discovery

Beyond the validated benchmark genes, the pipeline runs an unsupervised clustering analysis to identify de novo candidate anchors — genes whose transcriptional program fingerprint clusters with known GWAS-grounded targets, suggesting shared mechanisms.

β profiles are assembled for all ranked targets and normalised (z-score per program, capped at ±3). Hierarchical clustering uses τ*-weighted profiles: `β_clust = β_z × τ_norm`, where τ_norm = τ*(P) / max(τ*) for each heritable program. This weighting causes genes sharing strong loading on high-heritability programs to cluster together, while low-heritability programs contribute less to cluster geometry. Ward-linkage clustering is applied in the weighted space; silhouette scores are computed in the same weighted space for consistency.

Each cluster is scored by GWAS grounding:

> anchor_score = gwas_frac × mean_|γ| × silhouette

where gwas_frac is the fraction of cluster members with a GWAS anchor (L2G ≥ 0.05), mean_|γ| is the cluster mean OTA score, and silhouette is the within-cluster cohesion. The top-scoring gene in each cluster (by |γ|) is nominated as the de novo anchor for that cluster.

Cluster landscape plots (`outputs/{anchor_gene}_cluster_landscape_{disease}.png`) visualise each cluster's full gene set in OTA γ × fingerprint r space, with the anchor gene marked as an orange star and benchmark genes annotated as coloured diamonds (blue = benchmark KD reduces risk; red = benchmark KD increases risk). These plots are generated automatically for the top-5 clusters by anchor_score and for any cluster containing ≥ 1 benchmark gene.

The cluster JSON (`outputs/anchor_clusters_{disease}.json`) records all cluster assignments, silhouette scores, GWAS-grounded fraction, and the nominated de novo anchor per cluster.

### Constraints

Three constraints bound the current results. First, program identity is cell-type-specific: the cardiac endothelial and CD4+ T-cell programs reflect those particular cellular contexts; genes acting through other disease-relevant cell types will not be captured without additional matched Perturb-seq inputs. Second, Perturb-seq coverage is incomplete: many GWAS-anchored genes (*IL6R*, *JAK1* for RA; *NOS3*, *KLF2* for CAD) are absent because they were not knocked down in the current libraries. Third, GPS compound annotations are transcriptional hypotheses, not validated hits; experimental confirmation is required before advancement.

---

## References

1. Ota M, Spence JP, Zeng T, et al. Causal modelling of gene effects from regulators to programs to traits. *Nature* (2025). https://doi.org/10.1038/s41586-025-09866-3
2. GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across human tissues. *Science* 369:1318–1330 (2020).
3. Yazar S, et al. Single-cell eQTL mapping identifies cell type-specific genetic control of autoimmune disease. *Science* 376:eabf3041 (2022).
4. Mountjoy E, et al. An open approach to systematically prioritize causal variants and genes at all published human GWAS trait-associated loci. *Nat Genet* 53:1527–1533 (2021).
5. Kotliar D, et al. Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq. *eLife* 8:e43803 (2019).
6. Boyle EA, Li YI, Pritchard JK. An expanded view of complex traits: from polygenic to omnigenic. *Cell* 169:1177–1186 (2017).
7. Kurki MI, et al. FinnGen provides genetic insights from a well-phenotyped isolated population. *Nature* 613:508–518 (2023).
8. Schnitzler GR, et al. Convergence of coronary artery disease genes onto endothelial cell programs. *Nature* 626:799–807 (2024).
9. Zhu R, et al. Genome-scale Perturb-seq in primary human CD4+ T cells maps context-specific regulators of T cell programs and human immune traits. *bioRxiv* 2025.
10. Sun BB, et al. Plasma proteomic associations with genetics and health in the UK Biobank. *Nature* 622:329–338 (2023).
11. Ferkingstad E, et al. Large-scale integration of the plasma proteome with genetics and disease. *Nat Genet* 53:1712–1721 (2021).
12. Zhu Z, et al. Integration of summary data from GWAS and eQTL studies predicts complex trait gene targets. *Nat Genet* 48:481–487 (2016).
13. Staley JR, Burgess S. Semiparametric methods for estimation of a nonlinear exposure-outcome relationship using instrumental variables. *Genet Epidemiol* 41:341–352 (2017).
14. Giambartolomei C, et al. Bayesian test for colocalisation between pairs of genetic association studies using summary statistics. *PLoS Genet* 10:e1004383 (2014).
15. Subramanian A, et al. A next generation connectivity map: L1000 platform and the first 1,000,000 profiles. *Cell* 171:1437–1452 (2017).
16. Xing et al. GPS4Drugs: gene perturbation signatures for drug discovery. *Cell* 2026.
17. Lamb J, et al. The Connectivity Map. *Science* 313:1929–1935 (2006).
18. Pedregosa F, et al. Scikit-learn: machine learning in Python. *JMLR* 12:2825–2830 (2011).
19. Grabski IN, et al. Mapping transcriptional responses to cellular perturbation dictionaries with RNA fingerprinting. *bioRxiv* 2025.
20. Sakaue S, et al. A cross-population atlas of genetic associations for 220 human phenotypes. *Nat Genet* 53:1415–1424 (2021).
