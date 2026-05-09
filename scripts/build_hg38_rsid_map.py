"""
Build hg38 pos→rsID map by lifting the existing hg19 HapMap3 map.
Much faster than lifting 12M sumstats rows — lifts 1.19M map entries instead.
Output: data/ldsc/pos_rsid_map_hg38.tsv  (CHR POS_hg38 rsID)
"""
from pyliftover import LiftOver
from pathlib import Path

HG19_MAP = Path("data/ldsc/pos_rsid_map.tsv")
HG38_MAP = Path("data/ldsc/pos_rsid_map_hg38.tsv")

print("Loading hg38→hg19 liftover chain (auto-download if needed)...")
lo = LiftOver("hg19", "hg38")

print(f"Lifting {HG19_MAP} positions hg19→hg38...")
n_lifted = n_failed = 0

with open(HG19_MAP) as fin, open(HG38_MAP, "w") as fout:
    for line in fin:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        chrom, pos_hg19, rsid = parts[0], int(parts[1]), parts[2]
        result = lo.convert_coordinate(f"chr{chrom}", pos_hg19 - 1)  # 0-based for pyliftover
        if result:
            pos_hg38 = result[0][1] + 1  # back to 1-based
            fout.write(f"{chrom}\t{pos_hg38}\t{rsid}\n")
            n_lifted += 1
        else:
            n_failed += 1

print(f"  Lifted:  {n_lifted:,}")
print(f"  Failed:  {n_failed:,}")
print(f"  Output:  {HG38_MAP}")
