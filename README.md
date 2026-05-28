# circRNA LNP Delivery Prediction

A structure-function analysis of lipid nanoparticle (LNP) composition for nucleic acid delivery,
built on the LNPDB dataset (Collins et al., *Nat. Commun.* 2026; ~19,500 LNPs). The project
trains interpretable models (random forest, XGBoost) to predict transfection efficiency from
lipid composition, then asks which compositional features matter most for cargo properties
characteristic of **circular RNA** delivery — longer cargo length, higher secondary structure,
distinct physicochemical envelope. The circRNA angle is interpretive (no large public
circRNA-LNP dataset exists yet); the goal is to surface compositional hypotheses worth testing
experimentally.

## Status

In progress — see [`project1_plan.md`](../project1_plan.md) for the day-by-day plan.

## Reproduce

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Then run the notebooks in order:

1. `notebooks/01_eda.ipynb` — dataset exploration
2. `notebooks/02_features.ipynb` — feature engineering
3. `notebooks/03_modeling.ipynb` — model training + comparison
4. `notebooks/04_circrna_analysis.ipynb` — circRNA-adjacent interpretation

## Layout

```
data/         raw + processed datasets (raw/ gitignored)
notebooks/    analysis notebooks, numbered in execution order
src/          reusable Python modules (data loading, features, models)
outputs/      figures + saved models
writeup/      4–6 page report (markdown, exported to PDF)
```

## Data

Primary: **LNPDB** — https://lnpdb.molcube.com
Paper: https://www.nature.com/articles/s41467-026-68818-1

See [`data/README.md`](data/README.md) for column-level provenance once the data is pulled.

## Citation context

- Collins et al., *Nat. Commun.* 2026 — LNPDB
- Witten et al., *Nat. Biotech.* 2024 — AI-guided LNPs, pulmonary gene therapy
- Kulkarni et al., 2018 — LNP fundamentals
- Pardi et al., 2018 — mRNA therapeutics review
