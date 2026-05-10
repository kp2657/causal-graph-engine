"""
Pre-process GWAS sumstats for ldsc munge_sumstats.py.

Adds rsIDs from pos_rsid_map.tsv (CHR POS rsID) when needed (PLAtlas-style files).
GWAS Catalog harmonised files (Aragam 2022, Sakaue 2021) already have rsid column.

Usage:
  python scripts/munge_sumstats_for_ldsc.py cad       # Aragam 2022 GCST90132314
  python scripts/munge_sumstats_for_ldsc.py ra        # Sakaue 2021 GCST90132222
  python scripts/munge_sumstats_for_ldsc.py cad_old   # PLAtlas Phe_414 (legacy)
"""
import gzip
import sys
from pathlib import Path

LDSC_DIR = Path("data/ldsc")
POS_MAP  = LDSC_DIR / "pos_rsid_map.tsv"

CONFIGS = {
    "cad": {
        # Aragam 2022 (PMID 36474045) — rsid already in file, no pos_map needed
        "sumstats": LDSC_DIR / "sumstats" / "GCST90132314.h.tsv.gz",
        "out":      LDSC_DIR / "sumstats" / "CAD_ldsc_ready.sumstats.gz",
        "n":        1165690,  # 181,522 + 984,168
        "chr_col":  "chromosome", "pos_col": "base_pair_location",
        "snp_col":  "rsid",
        "a1_col":   "effect_allele", "a2_col": "other_allele",
        "beta_col": "beta", "se_col": "standard_error",
        "p_col":    "p_value",
        "use_pos_map": False,  # rsid already present
    },
    "ra": {
        # Sakaue 2021 (PMID 36333501) — rsid in file
        "sumstats": LDSC_DIR / "sumstats" / "GCST90132222.h.tsv.gz",
        "out":      LDSC_DIR / "sumstats" / "RA_ldsc_ready.sumstats.gz",
        "n":        None,   # N column present in file
        "chr_col":  "chromosome", "pos_col": "base_pair_location",
        "snp_col":  "rsid",
        "a1_col":   "effect_allele", "a2_col": "other_allele",
        "beta_col": "beta", "se_col": "standard_error",
        "p_col":    "p_value",
        "use_pos_map": False,
    },
    "cad_old": {
        # PLAtlas Phe_414 — needs pos_rsid_map.tsv (no rsID column)
        "sumstats": LDSC_DIR / "sumstats" / "Phe_414.EUR.gwama.sumstats.txt.gz",
        "out":      LDSC_DIR / "sumstats" / "CAD_ldsc_ready_platlas.sumstats.gz",
        "n":        818469,
        "chr_col":  "CHR", "pos_col": "POS",
        "snp_col":  None,
        "a1_col":   "ALT",  "a2_col": "REF",
        "beta_col": "BETA", "se_col": "SE",
        "p_col":    "P",
        "use_pos_map": True,
    },
}

disease = sys.argv[1].lower() if len(sys.argv) > 1 else "cad"
cfg = CONFIGS[disease]

pos_map: dict[tuple[str, str], str] = {}
if cfg.get("use_pos_map"):
    print(f"Loading pos→rsID map ({POS_MAP})...")
    with open(POS_MAP) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pos_map[(parts[0], parts[1])] = parts[2]
    print(f"  Loaded {len(pos_map):,} position→rsID entries")

print(f"Processing {cfg['sumstats']}...")
n_written = n_skipped_no_rsid = n_skipped_se = 0

with gzip.open(cfg["sumstats"], "rt") as fin, \
     gzip.open(cfg["out"], "wt") as fout:

    header = fin.readline().lstrip("#").strip().split("\t")
    col = {c: i for i, c in enumerate(header)}

    fout.write("SNP\tA1\tA2\tBETA\tSE\tP\tN\n")

    for line in fin:
        parts = line.strip().split("\t")
        if len(parts) < len(col):
            continue
        try:
            chrom = parts[col[cfg["chr_col"]]].lstrip("chr")
            pos   = parts[col[cfg["pos_col"]]]
            a1    = parts[col[cfg["a1_col"]]].upper()
            a2    = parts[col[cfg["a2_col"]]].upper()
            beta  = float(parts[col[cfg["beta_col"]]])
            se    = float(parts[col[cfg["se_col"]]])
            p     = float(parts[col[cfg["p_col"]]])
            n     = cfg["n"] if cfg["n"] else int(float(parts[col["n"]]))
        except (ValueError, KeyError, IndexError):
            n_skipped_se += 1
            continue

        if se <= 0:
            n_skipped_se += 1
            continue

        # Get rsID
        if cfg.get("use_pos_map"):
            rsid = pos_map.get((chrom, pos))
        else:
            rsid_col = cfg.get("snp_col", "rsid")
            rsid = parts[col[rsid_col]] if rsid_col in col else None
            if rsid in (None, "NA", ".", ""):
                rsid = None

        if rsid is None:
            n_skipped_no_rsid += 1
            continue

        # Skip indels for LDSC (SNPs only)
        if len(a1) > 1 or len(a2) > 1:
            n_skipped_no_rsid += 1
            continue

        fout.write(f"{rsid}\t{a1}\t{a2}\t{beta:.6f}\t{se:.6f}\t{p:.6g}\t{n}\n")
        n_written += 1

print(f"  Written:          {n_written:,}")
print(f"  No rsID / indel:  {n_skipped_no_rsid:,}")
print(f"  Bad SE:           {n_skipped_se:,}")
print(f"  Output:           {cfg['out']}")
