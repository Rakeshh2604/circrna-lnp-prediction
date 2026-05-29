# EDA summary — LNPDB cleaned dataset

- Cleaned dataset: **19,197 LNPs × 66 columns**
- Primary modeling subset (`luminescence_normalized`): **n = 14,302**
  - Unique ionizable lipids: 8,504
  - Unique publications: 31
  - Cargo split: mRNA 11,892, siRNA 1,330, pDNA 1,080

## Descriptor–target rank correlations (Spearman)

| Descriptor | ρ |
|---|---:|
| `rotatable_bonds` | +0.112 |
| `aromatic_rings` | -0.101 |
| `molar_refractivity` | +0.098 |
| `van_der_waals_molecular_volume` | +0.094 |
| `molecular_weight` | +0.094 |
| `heavy_atoms` | +0.093 |
| `rings` | -0.083 |
| `nitrogen_count` | +0.083 |
| `logp` | +0.083 |
| `fraction_sp3_carbons` | +0.057 |
| `topological_polar_surface_area` | +0.055 |
| `sp3_carbons` | +0.049 |
| `hydrogen_bond_acceptors` | +0.045 |
| `hydrogen_bond_donors` | +0.025 |

## Figures

- ![01_target_distribution](outputs/figures/eda/01_target_distribution.png)
- ![02_target_by_method](outputs/figures/eda/02_target_by_method.png)
- ![03_target_by_cargo](outputs/figures/eda/03_target_by_cargo.png)
- ![04_target_by_cell_line](outputs/figures/eda/04_target_by_cell_line.png)
- ![05_mol_pct_distributions](outputs/figures/eda/05_mol_pct_distributions.png)
- ![06_il_frequency](outputs/figures/eda/06_il_frequency.png)
- ![07_helper_peg_composition](outputs/figures/eda/07_helper_peg_composition.png)
- ![08_descriptor_corr_heatmap](outputs/figures/eda/08_descriptor_corr_heatmap.png)
- ![09_descriptor_vs_target](outputs/figures/eda/09_descriptor_vs_target.png)
- ![10_publication_sample_size](outputs/figures/eda/10_publication_sample_size.png)
- ![11_publication_target_variation](outputs/figures/eda/11_publication_target_variation.png)
- ![12_cell_x_cargo_heatmap](outputs/figures/eda/12_cell_x_cargo_heatmap.png)