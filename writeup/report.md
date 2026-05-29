# circRNA LNP Delivery Prediction — Report

*Draft. Sections 1, 3, 4, 5 to be filled in across Days 12 and 13.*

## 1. Introduction & Motivation

*To be written.*

## 2. Data

### 2.1 Source

The primary dataset is **LNPDB** (Collins et al., *Nat. Commun.* 2026), a public
structure-function compilation of lipid nanoparticle formulations curated from 42
peer-reviewed publications and one commercial supplier. The full CSV is freely
downloadable from https://lnpdb.molcube.com (no authentication required); we pulled
it on 2026-05-29.

Each row represents one LNP formulation tested in one biological context (one cell line
or animal model, one cargo, one assay). The raw file contains 19,797 rows and 66
columns, spanning four broad categories of information per LNP:

- **Composition** — identity, SMILES, and mol-percent for each of up to five lipid
  components (the ionizable lipid plus a helper lipid, cholesterol, PEG-lipid, and
  optional fifth component), plus ionizable-lipid sub-structural breakdown (head,
  linker, tails).
- **Formulation parameters** — aqueous buffer, dialysis buffer, mixing method,
  ionizable-lipid-to-nucleic-acid mass ratio.
- **Experimental context** — biological model (`in_vitro` or `in_vivo`), cell line or
  animal strain, route of administration, cargo class (mRNA / siRNA / pDNA), cargo
  identity (e.g. FLuc, hEPO, GFP), dose, assay type, and the measured value.
- **Pre-computed RDKit descriptors of the ionizable lipid** — heavy atoms, ring counts,
  rotatable bonds, van der Waals volume, topological polar surface area, hydrogen-bond
  donors and acceptors, LogP, molar refractivity, sp3-carbon fraction, nitrogen count,
  molecular weight, and binary indicators for ester, carbonate, and disulfide motifs.

The pre-computed descriptors are a meaningful convenience: they remove the need to
re-featurize lipids from SMILES with RDKit on our side, and they ensure consistent
featurization across the dataset.

### 2.2 Cleaning

We applied three filters to the raw data:

1. **Drop physical-characterization-only rows** (269 rows). These rows record the size,
   zeta potential, or hemolysis of an LNP but have no transfection measurement —
   identifiable by NaN values across the `Cargo`, `Cargo_type`, and `Model` columns.
   They are valid data for a separate physical-property analysis but irrelevant for
   transfection-efficiency prediction.
2. **Drop rows with a null `Experiment_value`** (328 additional rows). No target to
   train on.
3. **Drop rows whose mol-ratio components do not sum to 100%** (3 additional rows,
   tolerance `np.isclose`). These appear to be data-entry artifacts.

The cleaning pipeline lives in
[`src/data_loader.py`](../src/data_loader.py)::`clean_lnpdb` and reports a
`CleaningReport` object so the exact drop counts are reproducible. The cleaned dataset
contains **19,197 LNPs** (loss of 600 rows, ~3%) and is persisted as a Parquet file at
`data/processed/lnpdb_clean.parquet` (1.5 MB, 12× smaller than the source CSV).

### 2.3 Target variable choice

The `Experiment_value` column reports the assay readout — but `Experiment_method`
specifies which assay was used, and different methods are on different scales
(luminescence z-score, percent hemolysis, particle diameter in nm, zeta potential in
mV). Six methods are z-scored within their own population (mean 0, std 1), making them
comparable across studies; the remainder are physical or assay-specific scales.

We adopt **`luminescence_normalized`** as the primary modeling target — it is the
single largest measurement type (n = 14,302 rows after cleaning), z-scored to mean 0
and standard deviation 1, and is the readout most directly tied to transfection
efficiency. The closely related `luminescence_discretized_normalized` method
(n = 3,058) shares the same biological meaning and z-scored scale, so it can be
combined with the primary subset if additional sample size is needed for robustness
analyses. The primary subset spans 31 of the 42 source publications and contains
8,504 unique ionizable lipids — meaning each IL appears in roughly 1.7 LNPs on
average.

### 2.4 Key descriptive findings

**(i) Ionizable-lipid singletons dominate.** Of 12,431 unique ionizable lipids in the
cleaned dataset, 10,459 (84%) appear in exactly one LNP; another 1,812 appear in
between two and four LNPs. Only 4 ionizable lipids appear in 50 or more formulations.
This rules out categorical encoding of ionizable-lipid identity and motivates a
structural representation built from the 17 pre-computed RDKit descriptors.

**(ii) Formulations cluster around two canonical recipes.** The mol-percent
distributions for each of the four lipid components are bimodal, reflecting two
dominant formulation families: an "Onpattro-style" mode with ionizable-lipid mol%
near 50% and cholesterol near 38%, and a "Moderna/Pfizer-style" mode with ionizable
mol% near 35% and cholesterol near 46.5%. PEG-lipid is similarly bimodal at ~1.5%
and ~2.5%. The dataset is therefore not a uniform exploration of formulation
space — it is dense around two recipes and sparse elsewhere, which will limit how
far the model can extrapolate to novel composition envelopes.

**(iii) Cargo composition is mRNA-dominant and contains no circRNA.** The cleaned
dataset is 11,892 mRNA / 1,330 siRNA / 1,080 pDNA formulations on the primary
subset. There is no `circRNA` value in the `Cargo` column. The circRNA framing of
this project is therefore explicitly an *extrapolation*: we use mRNA as the closest
physicochemical analog (large, single-stranded, structured) and treat the model's
predictions on the mRNA subset — particularly the subset with high ionizable-lipid-to-
nucleic-acid mass ratios and large ionizable-lipid molecular weights — as a
defensible hypothesis source for circRNA delivery.

**(iv) No single chemistry descriptor strongly predicts efficiency.** Spearman rank
correlations between each of the 14 RDKit descriptors and the primary target are
uniformly weak (max |ρ| = 0.11, for `rotatable_bonds`). The sign pattern is
biophysically sensible — descriptors associated with more flexible, less rigid
lipids correlate positively with efficiency (`rotatable_bonds`, `fraction_sp3_carbons`,
`molecular_weight`, `nitrogen_count`), while descriptors of rigid aromatic content
correlate negatively (`aromatic_rings`, `rings`). The weakness of any single linear
relationship implies that whatever predictive signal exists is in *interactions* among
descriptors and across formulation context (cell line, cargo type, composition), and
that ensemble methods (random forest, XGBoost) are appropriate to learn those
interactions — but expectations for any model's $R^2$ should remain modest.

**(v) Study-level normalization preserves cross-study comparability.** Because the
target is z-scored within each `Experiment_method`, per-publication target medians
cluster tightly around zero across the top 15 publications. This is the intended
behavior of the dataset's normalization and means we do not need to add a
publication-level random effect at the target level — though we will still use
leave-one-publication-out (LOPO) cross-validation as the primary generalization
test (see Section 3).

### 2.5 Train/test design

The plan originally called for a leave-one-ionizable-lipid-out (LOLO) generalization
test. The IL-singleton dominance makes LOLO meaningless: holding out a lipid that
appears in only one row tells us nothing because that row is the only example. We
substitute **leave-one-publication-out (LOPO)** cross-validation across the 31
publications in the primary subset. LOPO asks the genuinely hard question: can the
model, trained on 30 published studies, predict efficiency for LNPs from a 31st study
it has never seen? This directly tests whether the model has learned generalizable
structure-function relationships or merely memorized study-specific patterns.

A standard 5-fold random split will serve as the easier baseline for model selection
and hyperparameter tuning, with LOPO as the final honesty check.

## 3. Methods

*To be written.*

## 4. Results

*To be written.*

## 5. Discussion & Limitations

*To be written.*

## 6. References

- Collins et al. (2026). *Nat. Commun.* — LNPDB. https://www.nature.com/articles/s41467-026-68818-1
- Witten et al. (2024). AI-guided LNPs for pulmonary gene therapy. *Nat. Biotechnol.*
- Kulkarni et al. (2018). LNP fundamentals.
- Pardi et al. (2018). mRNA therapeutics review.
