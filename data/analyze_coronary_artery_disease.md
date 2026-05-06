Causal graph analysis of coronary artery disease (EFO: EFO_0001645) identified 937 significant causal edges spanning 242 therapeutic targets. Top-ranked targets are ATP2B1, CCM2, IL6R. 162 gene–program β estimate(s) use the `provisional_virtual` evidence tier (in silico / weak perturbation support); other evidence streams may still be experimental.


## Target Rankings

| Rank | Gene | Ota γ | Tier | OT Score | Max Phase | Key Evidence |
|------|------|-------|------|----------|-----------|--------------|
| 1 | ATP2B1 | 18.136 | Tier1_Interventional | 0.50 | Phase 0 | — |
| 2 | CCM2 | -3.938 | Tier1_Interventional | 0.38 | Phase 0 | — |
| 3 | IL6R | 2.826 | Tier1_Interventional | 0.54 | Phase 4 | Drug: TOCILIZUMAB, VOBARILIZUMAB, LEVILIMAB |
| 4 | PROCR | 2.355 | Tier1_Interventional | 0.52 | Phase 0 | — |
| 5 | AGPAT4 | 2.218 | Tier1_Interventional | 0.50 | Phase 0 | — |
| 6 | DHX36 | -2.175 | Tier1_Interventional | 0.33 | Phase 0 | — |
| 7 | HMGCR | -2.150 | Tier1_Interventional | 0.73 | Phase 4 | Drug: ROSUVASTATIN, PITAVASTATIN MAGNESIUM, CERIVASTATIN |
| 8 | TGFB1 | -1.857 | Tier1_Interventional | 0.47 | Phase 4 | Drug: FRESOLIMUMAB, METELIMUMAB, LY-2382770 |
| 9 | MCL1 | -1.753 | Tier1_Interventional | 0.34 | Phase 3 | Drug: OBATOCLAX MESYLATE, OBATOCLAX |
| 10 | BCAS3 | -1.700 | Tier1_Interventional | 0.40 | Phase 0 | — |


## Top Target Narratives

ATP2B1 upregulates disease-relevant CAD_SVD_C07, CAD_SVD_C11, CAD_SVD_C18 → disease trait.
Evidence: GWAS + Perturb-seq.
Composite Ota estimate: γ_{ATP2B1→trait} = 18.136 (Tier1_Interventional).
Best compound: no approved drug (no in vitro activity data; Phase 0).

CCM2 downregulates disease-promoting CAD_SVD_C07, CAD_SVD_C11, CAD_SVD_C18 → disease trait.
Evidence: GWAS + Perturb-seq.
Composite Ota estimate: γ_{CCM2→trait} = -3.938 (Tier1_Interventional).
Best compound: no approved drug (no in vitro activity data; Phase 0).

IL6R upregulates disease-relevant CAD_SVD_C07, CAD_SVD_C18, CAD_SVD_C12 → disease trait.
Evidence: Drug: TOCILIZUMAB, VOBARILIZUMAB, LEVILIMAB.
Composite Ota estimate: γ_{IL6R→trait} = 2.826 (Tier1_Interventional).
Best compound: TOCILIZUMAB (no in vitro activity data; Phase 4).


## Limitations

About 162 gene–program β rows are labelled `provisional_virtual` (in silico or weakly supported cell-model estimates). That label applies to those β edges, not to the entire analysis: GWAS/L2G and other tiers may still be experimentally grounded. Treat virtual-tier rows as hypothesis-generating. Full quantitative Perturb-seq β coverage may require local GEO GSE246756 downloads (~50GB; Replogle 2022) when not cached. Program γ estimates are context-dependent; validate lead findings experimentally.