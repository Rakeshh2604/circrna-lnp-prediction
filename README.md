# circRNA LNP Delivery Prediction

A structure–function analysis of lipid nanoparticle (LNP) composition for nucleic-acid
delivery, built on the LNPDB dataset (Collins et al., *Nat. Commun.* 2026; ~19,800
LNPs across 42 publications). The project trains interpretable models on the full
dataset and uses TreeSHAP to ask **which compositional features drive transfection
efficiency** — and how that answer shifts in a sub-regime that approximates **circRNA**
delivery requirements.

The headline finding is methodological: under **leave-one-publication-out** cross-
validation, the model's R² drops from +0.33 to −0.11, meaning most of what the model
learns under random splits is study-level memorization rather than portable
structure–function signal. The interpretation results should be read as
**hypothesis generation**, not as a usable cross-study predictor.

## Headline results

| | Random 5-fold CV R² | LOPO R² |
|---|---:|---:|
| Mean baseline | 0.00 | — |
| Ridge (L2) | +0.05 | — |
| Random forest | +0.31 | — |
| **XGBoost (tuned)** | **+0.33** | **−0.11** |

- **Dominant lever overall:** ionizable-lipid chemistry (`rotatable_bonds`, TPSA,
  `nitrogen_count`, `logp`) — group-aggregated mean |SHAP| of 0.55, three times the
  next group (composition mol%, 0.18).
- **circRNA-like regime** (mRNA cargo, IL:NA mass ratio > 10, n = 2,096): the feature
  hierarchy shifts. **PEG mol%** becomes the #1 lever (was #3) and the **IL:NA mass
  ratio** jumps from #8 to #2. Hypothesis: when designing LNPs for long structured
  cargoes like circRNA, prioritize PEG composition and IL loading first, chemistry
  second.

Full discussion: [`writeup/report.md`](writeup/report.md). Slide deck:
[`slides/deck.md`](slides/deck.md).

## Reproduce from scratch

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Pull the raw data manually
#    From https://lnpdb.molcube.com -> Downloads -> "Download LNPDB.csv"
#    Save it to data/raw/LNPDB.csv

# 3. Run the pipeline end-to-end
python -m src.data_loader              # 19,797 -> 19,197 rows; saves cleaned parquet
python -m src.features                 # builds X.parquet + y.parquet + feature_meta.parquet
python notebooks/01_eda.py             # 12 EDA figures + summary
python notebooks/03_modeling.py        # mean baseline / Ridge / RF; 5-fold CV
python notebooks/04_xgboost.py         # tunes + evaluates XGBoost
python notebooks/05_shap.py            # SHAP values + interpretation figures
python notebooks/06_lopo.py            # leave-one-publication-out
python notebooks/07_circrna_cut.py     # circRNA-adjacent SHAP analysis
```

Everything regenerates deterministically (`random_state=42` throughout) on a fresh
clone provided you place `data/raw/LNPDB.csv` first. Total runtime end-to-end is ~10
minutes on a 2020 MacBook Air.

## Layout

```
data/
  raw/                              not in git; place LNPDB.csv here
  processed/                        lnpdb_clean.parquet + X/y/feature_meta
  README.md                         column-by-column schema documentation

src/
  data_loader.py                    load_lnpdb_raw / clean_lnpdb / save+load_processed
  features.py                       build_features (X, y, FeatureSpec) + save/load
  models.py                         CVResult dataclass + cv_evaluate harness

notebooks/                          standalone Python scripts (not Jupyter; portable)
  01_eda.py                         12 exploratory figures + summary
  03_modeling.py                    mean baseline / Ridge / Random Forest
  04_xgboost.py                     RandomizedSearchCV-tuned XGBoost
  05_shap.py                        TreeSHAP + group-aggregated importance
  06_lopo.py                        leave-one-publication-out CV
  07_circrna_cut.py                 mRNA + high-mass-ratio subset SHAP

outputs/
  metrics_baseline.csv              all 4 models' RMSE/R²
  oof_predictions.parquet           out-of-fold preds for every model
  best_xgboost_params.json
  shap_values.parquet               TreeSHAP values for all 14,302 rows
  lopo_per_publication.csv          per-publication LOPO RMSE/R²
  eda_summary.md                    structured EDA findings
  circrna_cut_summary.md            circRNA-cut summary
  figures/
    eda/                            12 figs
    modeling/                       5 figs (incl. XGBoost)
    shap/                           8 figs
    lopo/                           2 figs
    circrna/                        3 figs

slides/deck.md                      11-slide Marp deck + 3 backup slides
writeup/
  report.md                         ~5-6 page report (6 sections, references)
  notes.md                          paper + dataset notes
```

## Data sources

- **LNPDB** — https://lnpdb.molcube.com
  Collins et al. (2026). *Nat. Commun.* https://www.nature.com/articles/s41467-026-68818-1

## Limitations (read these before citing the model's predictions)

1. **No circRNA in the data.** The "circRNA cut" is interpretive, not validated.
2. **Cross-study generalization is poor** (LOPO mean R² = −0.11). Within-study
   predictions are meaningfully above baseline; cross-study predictions are not.
3. **Recipe clustering** — the dataset is dense around two canonical formulation
   families (Onpattro-style and Moderna/Pfizer-style) and sparse elsewhere.
4. **Tree-ensemble mean regression** under-predicts the rare high-performing tail —
   exactly the LNPs a screening pipeline would most want to identify.

See `writeup/report.md` Section 5 for a fuller treatment.

## License

The code in this repository is released for academic use. The LNPDB dataset retains
its own licensing — please consult the LNPDB site and the Collins et al. 2026 paper.
