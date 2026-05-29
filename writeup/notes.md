# Notes

## Day 1 — paper read (2026-05-29)

Collins et al., *Nat. Commun.* 2026 — LNPDB introduction paper.
Read by Rakesh. Detailed scientific notes to be filled in here from his understanding.

## Day 2 — first look at the data (2026-05-29)

`raw/LNPDB.csv` pulled directly from https://lnpdb.molcube.com/Downloads. 19,797 rows × 66 cols.
Full column-by-column documentation lives in [`../data/README.md`](../data/README.md).

### Things that surprised us vs. the plan

1. **Pre-computed RDKit descriptors are included.** Columns 50–66 (Heavy.Atoms, LogP, TPSA,
   MW, etc.) are already in the CSV. Day 6's "compute RDKit descriptors" step is unnecessary;
   we use what's there.

2. **There is no `circRNA` value in the `Cargo` column.** Only mRNA / siRNA / pDNA. The plan
   already flagged this risk — the circRNA framing must be interpretive, not data-supported.
   The closest physicochemical neighbor in the dataset is mRNA (long, single-stranded,
   structured), so the circRNA cut should be built on the mRNA subset filtered for high
   `IL_to_nucleicacid_massratio` and the upper tail of IL molecular weight (proxy for the
   larger LNPs that long cargoes need).

3. **Ionizable lipid is nearly always unique per row.** 13,028 unique ILs across ~19,500 rows.
   The plan's "one-hot encode `IL_name`" approach is dead. The ionizable lipid must be
   represented by structural features (the 17 RDKit columns) — not categorical identity.
   This is the single most important modeling decision and it changes Day 6's plan.

4. **Target is heterogeneous across measurement methods.** `Experiment_value` is on different
   scales for different `Experiment_method` values. Six of the methods are z-scored within
   method (mean ≈ 0, std = 1) — these are comparable across studies. Primary modeling target
   should be `luminescence_normalized` (n = 14,302 with non-null target).

### Adjustments to the project plan

- **Day 3 (cleaning) is mostly done already** — the CSV is clean. Remaining work: drop the
  269 NA-heavy physical-characterization rows for the predictive model, drop the 597 rows
  with NA `Experiment_value`, drop the ~272 rows with mol% sums ≠ 100, save as parquet.
  Realistically ~2 hours, not 5–6.
- **Day 4 (EDA)** — still useful for distributions, target stratification by cell line,
  publication-level effect sizes, IL-feature correlations with efficiency.
- **Day 6 (feature engineering)** — replace "one-hot encode ionizable lipid identity" with
  "use the 17 pre-computed RDKit descriptors as the IL representation." One-hot helper /
  chol / PEG (7 / 16 / 15 unique each) — these stay as planned. Add mol% numerics.
- **Day 10 (robustness)** — LOLO (leave-one-IL-out) is meaningless when ILs are mostly
  singletons. Replace with **LOPO (leave-one-publication-out)**: 42 publications, holding one
  out at a time gives a genuine "can this model generalize to a new lab/study?" test.
  This is a stronger and more honest robustness check anyway.
