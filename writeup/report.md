# circRNA LNP Delivery Prediction — Report

*Project 1 of 4 in a graduate-application portfolio. ~3 weeks of work, condensed
into a single end-to-end pass. Author: Rakesh Vayigandla, Northeastern.*

## 1. Introduction & Motivation

Lipid nanoparticles (LNPs) are the dominant delivery vehicle for nucleic acid
therapeutics. Onpattro (siRNA, 2018), the BNT162b2 and mRNA-1273 COVID-19 vaccines
(2020), and a growing pipeline of mRNA, base-editor, and protein-replacement therapies
all rely on the four-component LNP chassis: an ionizable lipid (the workhorse of
endosomal escape), a helper lipid (membrane fluidity), cholesterol (membrane
stability), and a PEG-lipid (circulation half-life). Despite the chassis being shared,
each new cargo class — siRNA, mRNA, pDNA, base editors, gene editors — has historically
required its own composition screen, often involving thousands of formulations, to find
the right balance of those four components for that cargo's biophysical profile.

Circular RNA (circRNA) is a candidate next-generation cargo. It is more stable than
linear mRNA (resistant to exonucleases due to its covalent loop), can be designed to
encode arbitrary proteins, and has shown promising delivery characteristics in early
preclinical work. But there is no large public dataset that pairs LNP composition with
circRNA delivery efficiency, which means rational design of circRNA-targeted LNPs
currently depends on transferring intuitions from the linear-mRNA literature without
quantitative grounding.

This project asks two questions on the largest public LNP composition/efficiency
dataset — **LNPDB** (Collins et al., *Nat. Commun.* 2026; ~19,800 formulations across
42 publications):

1. **Within the available data, which compositional features most strongly drive
   transfection efficiency, and how does the answer shift in a sub-regime that
   approximates circRNA delivery requirements?** We use the mRNA subset filtered to
   high ionizable-lipid-to-nucleic-acid mass ratios as that proxy regime.
2. **How well do those findings generalize across studies?** Random-split cross-
   validation is the field's standard but routinely overstates real-world predictive
   power because it doesn't separate signal from study-level confounding. We test this
   directly with leave-one-publication-out (LOPO) cross-validation.

The intended contribution is a defensible set of hypotheses about circRNA-LNP design
levers, together with an honest accounting of how much we should trust them given
the cross-study generalization gap.

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
interactions — but expectations for any model's R² should remain modest.

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

### 3.1 Feature engineering

We build a 57-feature design matrix (`src/features.py`) on the cleaned primary subset
(n = 14,302). Features are grouped semantically for downstream interpretability:

| Group | n features | Source |
|---|---:|---|
| Composition (mol%) | 4 | `il_molratio`, `hl_molratio`, `chl_molratio`, `peg_molratio` |
| Formulation numeric | 1 | `il_to_nucleicacid_massratio` |
| IL chemistry — numeric | 14 | Pre-computed RDKit descriptors |
| IL chemistry — binary | 3 | `has_ester`, `has_carbonate`, `has_disulfide` |
| Helper lipid identity | 7 | One-hot of 6 helper lipids + "(none)" |
| Cargo class | 3 | One-hot: mRNA, siRNA, pDNA |
| Biological model (broad) | 2 | One-hot: in_vitro, in_vivo |
| Biological model (specific) | 14 | One-hot of cell lines and mouse strains |
| Mixing method | 2 | handmixed vs microfluidics |
| Aqueous buffer | 3 | citrate, acetate, Unknown |
| PEG-lipid identity (top + pooled) | 4 | DMG-PEG2000, DMPE-PEG2000, (none), (other) |

We deliberately exclude `dose_ug_nucleicacid`, `publication_pmid`, `publication_link`,
and `cargo_type` as study-leakage proxies that would inflate random-CV performance
without representing transferable structure-function signal. Cholesterol identity
(`chl_name`) is dropped because 99.8% of rows use standard cholesterol — no variance
to learn from. The design matrix has no missing values after the cleaning pipeline.

### 3.2 Models

Three model families, all trained on the same `(X, y)`:

1. **Mean baseline** (`DummyRegressor(strategy="mean")`) — a sanity check; gives RMSE
   ≈ 1.0 by construction because y is z-scored.
2. **L2-regularized linear regression** (`RidgeCV` over α ∈ {0.01, 0.1, 1, 10, 100})
   — establishes whether any meaningful signal is captured by linear effects.
3. **Random forest** (`RandomForestRegressor`, n_estimators=300, min_samples_leaf=5)
   and **XGBoost** (`XGBRegressor`, tuned via `RandomizedSearchCV` over 30 candidates
   with a 3-fold inner CV, scored on negative RMSE; the search space covers
   learning_rate ∈ log-uniform[0.01, 0.3], max_depth ∈ [3, 9], min_child_weight ∈
   [1, 9], subsample/colsample_bytree ∈ [0.6, 1.0], and reg_alpha / reg_lambda ∈
   log-uniform ranges).

### 3.3 Cross-validation protocols

Two protocols, run on the same tuned XGBoost so the comparison is clean:

- **Random 5-fold CV** (the standard) with shuffled splits, fixed seed 42. Used for
  hyperparameter selection and for the headline model-comparison plots.
- **Leave-one-publication-out (LOPO)** across the 29 publications in the primary
  subset that have ≥ 30 LNPs. For each held-out publication, we refit XGBoost on all
  other rows (including the two smaller publications) and report RMSE and R² on the
  held-out set. This is the project's honest generalization metric.

### 3.4 Interpretation

We compute **TreeSHAP** values for the tuned XGBoost on the full primary subset
(`src/05_shap.py` and `outputs/shap_values.parquet`). For each feature we report
mean |SHAP| as global importance, and we aggregate by the feature groups in §3.1 to
identify which categories of features drive predictions overall and within subsets.

The **circRNA-adjacent analysis** (§4.4) restricts to LNPs with `cargo == "mRNA"` and
`il_to_nucleicacid_massratio > 10` — the latter threshold isolates the ~18% of mRNA
formulations where the curator chose to load more ionizable lipid than the
dataset-standard value of 10, a proxy for the demanding-delivery regime that long
structured cargoes like circRNA would impose. We recompute group- and feature-level
SHAP summaries on this subset and compare to the overall ranking.

## 4. Results

### 4.1 Random 5-fold CV: ensemble methods sharply outperform linear

| Model | RMSE | R² | Notes |
|---|---:|---:|---|
| Mean baseline | 0.9997 ± 0.008 | −0.0002 ± 0.000 | sanity |
| Ridge (L2) | 0.9731 ± 0.015 | +0.052 ± 0.016 | linear signal is weak |
| Random forest | 0.8304 ± 0.011 | +0.310 ± 0.012 | ~6× lift over Ridge |
| **XGBoost (tuned)** | **0.8186 ± 0.011** | **+0.329 ± 0.011** | best random-CV |

The 6× lift between Ridge and the tree models confirms what the EDA suggested
(Section 2.4 (iv)): no single descriptor has a strong linear association with the
target, so the predictive signal lives in interactions. XGBoost's edge over the random
forest (~2 percentage points of R²) is modest but consistent across folds.

The random-forest residual plot ([fig](../outputs/figures/modeling/03_residuals_rf.png))
shows a real systematic bias: the model under-predicts the rare high-performing
outliers, with predictions capping near +3 even though actual values reach +10. This
is classic mean-regression behavior for tree ensembles and matters because the high-
performers are exactly the LNPs a screening pipeline would want to identify.

### 4.2 LOPO: the model does not generalize across studies

| Metric | Random 5-fold CV | LOPO |
|---|---:|---:|
| Mean R² | +0.329 | **−0.108** |
| Median R² | (n/a) | −0.056 |
| Worst held-out R² | (n/a) | −0.935 (PMID 20080679) |
| Best held-out R² | (n/a) | +0.298 (PMID 40060499) |
| Publications with R² > 0 | (n/a) | 8 / 29 |
| Publications with R² > 0.1 | (n/a) | 3 / 29 |

The 0.44-point R² gap between random CV and LOPO is the project's most important
result. It says that most of what the model is doing in the random-CV setting is
*memorizing study-level idiosyncrasies* — the specific ionizable-lipid family a lab
favors, the particular mixing protocol, the cell-line/cargo combinations unique to a
given paper — rather than learning portable structure-function signal. Of the 29
publications evaluated under LOPO, only 8 yield positive held-out R² and only 3
exceed R² = 0.10. The worst publication (PMID 20080679, an older siRNA screen) is
predicted worse than a constant-mean predictor by a substantial margin.

This is the failure mode the project plan flagged in advance ("if R² is low on
held-out data, this is informative, not a failure — report honestly"). It is also
a known pathology in ML-on-biology more broadly: random splits do not separate
biological signal from study-specific confounding, and most published models that
report only random-CV performance probably suffer from this exact issue to some
degree.

### 4.3 SHAP: ionizable-lipid chemistry is the dominant lever

[Figure: outputs/figures/shap/03_shap_bar_by_group.png] — group-aggregated mean
|SHAP| from the tuned XGBoost. The ranking is unambiguous:

```
il_chemistry_numeric  (0.55)  >>  composition  (0.18)  >  hl_name  (0.06)
                                                       ≈  model_type (cell line)  (0.05)
                                                       >  formulation_numeric  (0.04)
                                                       >  cargo  (0.02)
                                                       >  peg_name / il_chemistry_binary  (~0.01)
                                                       >  model / aqueous_buffer / mixing_method  (~0)
```

Within IL chemistry, the top individual features are `rotatable_bonds` (mean |SHAP|
0.11), `topological_polar_surface_area` (0.07), `nitrogen_count` (0.07), `logp`
(0.07), and `fraction_sp3_carbons` (0.04). The biophysical reading is consistent:
ionizable lipids that are **more flexible, more polar/charged, and more aliphatic**
deliver better, on average. The only non-IL-chemistry feature to crack the top 10 is
`peg_molratio` (#3, 0.07) and `hl_molratio` (#6, 0.06) — composition mol% matters,
but secondary to lipid chemistry.

The negative finding is informative too: cell line (`model_type` group, 0.05) and
cargo class (`cargo`, 0.02) contribute relatively little to predictions, which means
the model has learned a representation in which the same lipid behaves similarly
across cells and across mRNA/siRNA/pDNA — modulo the LOPO caveat, which suggests
that this apparent invariance is partly an artifact of within-study training.

### 4.4 circRNA-adjacent analysis: the feature hierarchy shifts

Restricting to mRNA cargo with IL:NA mass ratio > 10 yields n = 2,096 LNPs in the
"circRNA-like" demanding-delivery subset. Group-aggregated SHAP on this subset versus
overall:

| Feature group | All LNPs | circRNA-like | Δ |
|---|---:|---:|---|
| IL chemistry numeric | 0.59 | 0.48 | ↓ |
| Composition (mol%) | 0.18 | 0.25 | ↑ |
| Formulation (IL:NA ratio) | 0.04 | **0.10** | **↑ 2.5×** |
| Helper lipid identity | 0.06 | 0.07 | slight ↑ |

Top-5 features by mean |SHAP|, within subset:

| Rank | All LNPs | circRNA-like (mRNA, IL:NA > 10) |
|---:|---|---|
| 1 | `rotatable_bonds` (0.11) | **`peg_molratio` (0.12)** |
| 2 | `topological_polar_surface_area` (0.07) | **`il_to_nucleicacid_massratio` (0.11)** |
| 3 | `peg_molratio` (0.07) | `rotatable_bonds` (0.10) |
| 4 | `nitrogen_count` (0.07) | `topological_polar_surface_area` (0.08) |
| 5 | `logp` (0.07) | `hl_molratio` (0.08) |

In the demanding-delivery regime, **PEG mol% becomes the #1 lever** (was #3) and the
**IL:NA mass ratio jumps from #8 to #2**. Predicted-efficiency envelopes shift
upward: median OOF prediction is +0.079 (vs −0.037 overall) and the 90th-percentile
prediction is +0.895 (vs +0.656 overall), a ~38% increase in the predicted ceiling.

The model's hypothesis for circRNA-class delivery is therefore: **prioritize tuning
PEG composition and the IL-to-nucleic-acid loading ratio first, then chemistry.**
This is a meaningfully different design prescription than the overall dataset would
suggest for typical mRNA delivery, where chemistry leads. It is consistent with
biophysical intuition — in challenging delivery, the PEG corona (which controls
circulation, stability, and endosomal escape kinetics) becomes more decisive, and
the formulator's ability to load extra ionizable lipid is the most direct way to
compensate for a larger or more structured cargo.

## 5. Discussion & Limitations

### 5.1 What this work shows and does not show

This project does **not** demonstrate that the resulting model can predict the
efficiency of an LNP for a cargo class it has never seen. The LOPO result makes that
clear: even within mRNA, the model fails to transfer across study boundaries. What
this work **does** show is two things. First, on a within-study basis, ionizable-
lipid chemistry — particularly flexibility, polar surface area, nitrogen content, and
hydrophobicity — is the dominant feature group driving the predictions of a tuned
XGBoost on LNPDB; secondary contributions come from composition mol% (especially PEG
and helper lipid), with cell line, cargo, buffer, and mixing method contributing
substantially less. Second, when we condition on the demanding-delivery regime that
a long structured cargo like circRNA would impose, the feature hierarchy *shifts in
a specific, biophysically plausible way*: PEG mol% and IL loading become the top two
levers. These two findings, taken together, are this project's hypothesis-generating
contribution.

### 5.2 Sources of the LOPO gap

The 0.44-point gap between random-CV and LOPO R² is the central limitation. Several
non-exclusive explanations are likely:

- **Composition recipe clustering** (Section 2.4 (ii)): the dataset is dense around
  two canonical formulations and sparse elsewhere. A held-out publication may use a
  composition that the model has effectively never been trained on.
- **Lab-specific ionizable lipid libraries**: 84% of ionizable lipids are singletons,
  and they cluster by paper — held-out publications often contain entire structural
  families absent from training.
- **Study-level assay/normalization differences** that the within-method z-score does
  not fully absorb (e.g., different cell-line passages, different luciferase
  substrates, different timing).

The honest framing for the writeup and downstream lab outreach is that random-CV R²
is *not* the headline number; LOPO is. The model is currently usable as a
hypothesis-generation tool inside an experimental envelope similar to the training
distribution, not as a black-box predictor on a novel chemistry or formulation.

### 5.3 Other limitations

- **No circRNA data.** The circRNA framing is interpretive throughout. The
  "circRNA-like" subset (mRNA + IL:NA mass ratio > 10) is a proxy and not a substitute
  for actual circRNA delivery measurements.
- **Tree-ensemble mean regression on the high-performer tail**: predictions cap near
  +3 even when actual values reach +10 (Section 4.1). The top 1% of LNPs in the data
  are precisely the ones a real screening pipeline would most want to identify, and
  the model is systematically under-confident on them. This affects the predicted-
  envelope numbers in §4.4: the +38% upper-tail lift in the circRNA-like subset is
  almost certainly a conservative estimate of the true ceiling.
- **Target heterogeneity**: even after restricting to `luminescence_normalized`, the
  exact cell line, luciferase substrate, time point, and imaging protocol vary across
  the 31 publications. Within-method z-scoring reduces but does not eliminate this.

### 5.4 What would meaningfully improve this

In rough order of likely impact:

1. **A LOPO-aware retraining objective** — e.g. group-aware gradient boosting,
   domain-adversarial training, or invariant risk minimization — to discourage the
   model from leaning on publication-identifying features.
2. **Restricting to the within-recipe envelope** and reporting performance separately
   for the Onpattro family vs the Moderna/Pfizer family; the model probably
   generalizes much better within a family than across.
3. **Acquiring even ~100 LNPs of real circRNA-delivery data** (from a collaborator or
   commercial supplier) would convert the entire "extrapolation" framing into a
   genuine fine-tuning study and would be far more directly useful for the Anderson
   lab's circRNA work than any further refinement of the LNPDB-only model.
4. **Adding cargo sequence-level features** (length, GC content, secondary-structure
   metrics) once those become available — this is the dimension along which "more
   like circRNA" would actually be measurable.

### 5.5 Closing

The most honest one-sentence summary of this work: *LNPDB is a useful dataset for
generating composition hypotheses about LNP delivery, the model trained on it
identifies ionizable-lipid chemistry as the dominant within-study lever and PEG /
IL-loading as the top circRNA-regime levers, and the cross-study generalization gap
is wide enough that any of those hypotheses needs experimental validation before
being trusted.*

## 6. References

- Collins et al. (2026). *Nat. Commun.* — LNPDB. https://www.nature.com/articles/s41467-026-68818-1
- Witten et al. (2024). AI-guided LNPs for pulmonary gene therapy. *Nat. Biotechnol.*
- Kulkarni et al. (2018). On the formation and morphology of lipid nanoparticles
  containing ionizable cationic lipids and siRNA. *ACS Nano* 12:4787–4795.
- Pardi et al. (2018). mRNA vaccines — a new era in vaccinology. *Nat. Rev. Drug
  Discov.* 17:261–279.
- Hajj & Whitehead (2017). Tools for translation: non-viral materials for therapeutic
  mRNA delivery. *Nat. Rev. Materials* 2:17056.
- Lundberg & Lee (2017). A unified approach to interpreting model predictions.
  *NeurIPS* — TreeSHAP.
- Chen & Guestrin (2016). XGBoost: A scalable tree boosting system. *KDD*.
