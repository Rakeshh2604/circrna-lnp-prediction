# Data

## Sources

### LNPDB (primary)
- URL: https://lnpdb.molcube.com
- Paper: Collins et al., *Nat. Commun.* 2026 — https://www.nature.com/articles/s41467-026-68818-1
- Contents: ~19,528 lipid nanoparticles with composition (ionizable lipid, helper lipid,
  cholesterol, PEG-lipid; mole percentages), experimental method, and functional results.
- Access pulled on: TBD (fill in once downloaded)
- Local path: `raw/lnpdb_full.csv` (or whatever the actual download name is)

### Witten et al. 2024 (secondary, optional)
- Paper: *Nat. Biotech.* 2024, AI-guided LNPs for pulmonary gene therapy
- Use: supplementary tables may have a composition–efficiency table; check after primary
  dataset is set up.

## Columns

Filled in after Day 3 (cleaning). Document units and meaning per column once the schema is
confirmed against the paper's Methods section.

## Provenance notes

- `raw/` is gitignored; treat as a download cache, regenerable from upstream.
- `processed/` holds the cleaned parquet produced by `src/data_loader.py::clean_lnpdb`.
