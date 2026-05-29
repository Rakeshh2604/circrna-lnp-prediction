# Data

## Source

**LNPDB** — https://lnpdb.molcube.com
Paper: Collins et al., *Nat. Commun.* 2026 — https://www.nature.com/articles/s41467-026-68818-1

The full dataset is freely downloadable from the Downloads page as a single CSV (`LNPDB.csv`), no login required.

- File: `raw/LNPDB.csv` (18 MB)
- Downloaded: 2026-05-29
- Rows: 19,797
- Columns: 66
- Memory loaded as pandas: ~54 MB

Site reports 19,797 LNP formulations, 12,844 unique ionizable lipids, 43 publications + 1 commercial supplier. Our inspection finds 13,028 unique `IL_name` values — slight discrepancy, likely due to alternate naming conventions for the same lipid.

## Schema

66 columns, grouped:

### Identifiers (4)
| Col | Meaning |
|---|---|
| `Index` | Row index (1-based) |
| `LNP_ID` | Stable LNP identifier (e.g., `LNP_0000001`) |
| `Experiment_ID` | Experiment cluster identifier |
| `Formulation_ID` | Formulation identifier |

### Ionizable lipid — structure (15)
- `IL_name`, `IL_SMILES`, `IL_protonated_SMILES`
- `IL_head_name`, `IL_head_SMILES`, `IL_linker_name`, `IL_linker_SMILES`
- `IL_tail1_name`, `IL_tail1_SMILES`, `IL_tail2_name`, `IL_tail2_SMILES`, `IL_tail3_name`, `IL_tail3_SMILES`, `IL_tail4_name`, `IL_tail4_SMILES`

Tail3 / tail4 are 100% NA — only some lipids have 3–4 tails. Tail2 is NA for 15,877 rows.

### Ionizable lipid — composition (3)
- `IL_molratio` (mol % of the IL component in the LNP)
- `IL_to_nucleicacid_massratio`
- `IL_to_nucleicacid_chargeratio` (98% NA — sparsely populated)

### Helper lipid (3)
`HL_name`, `HL_SMILES`, `HL_molratio`. 7 unique helper lipids. 838 rows have no helper lipid (HL_name NA).

### Cholesterol (3)
`CHL_name`, `CHL_SMILES`, `CHL_molratio`. 16 unique cholesterols.

### PEG-lipid (3)
`PEG_name`, `PEG_SMILES`, `PEG_molratio`. 15 unique PEG-lipids.

### Fifth component (3)
`fifthcomponent_name`, `fifthcomponent_SMILES`, `fifthcomponent_molratio`. 99.8% NA — rarely used.

### Formulation (3)
- `Aqueous_buffer` (e.g., acetate)
- `Dialysis_buffer` (mostly NA)
- `Mixing_method` (2 unique values, including `handmixed`)

### Biological model (4)
- `Model` — `in_vitro` / `in_vivo` (broad category)
- `Model_type` — specific cell line (HeLa, A549, HepG2, DC2.4, RAW264.7, IGROV1, HEK293T, ...) or mouse strain (Mouse_B6, Mouse_BALBc, Mouse_ICR, Mouse_CD1, Mouse_Ai14)
- `Model_target` — secondary descriptor
- `Route_of_administration` — `in_vitro` for cell culture; `intravenous`, `intramuscular`, `intradermal`, `intratracheal` for in vivo

### Cargo (3)
- **`Cargo`** — molecular class: `mRNA` (14,578), `siRNA` (3,758), `pDNA` (1,192). **No circRNA.**
- `Cargo_type` — protein/reporter encoded: `FLuc` (17,752), `DNA_barcode`, `peptide_barcode`, `hEPO`, `base_editor`, `FVII`, `GFP`, `RLuc`
- `Dose_ug_nucleicacid` — administered dose

### Experimental measurement (3)
- `Experiment_method` — assay type, defines the unit/scale of `Experiment_value`
- `Experiment_batching` — `individual` or batched
- **`Experiment_value`** — the target variable (units depend on method)

### Publication (2)
- `Publication_link` — URL to the source paper
- `Publication_PMID` — PubMed ID. 42 unique publications.

### Pre-computed RDKit descriptors of the ionizable lipid (17)
`Heavy.Atoms`, `Rings`, `Aromatic.Rings`, `Rotatable.Bonds`, `van.der.Waals.Molecular.Volume`, `Topological.Polar.Surface.Area`, `Hydrogen.Bond.Donors`, `Hydrogen.Bond.Acceptors`, `LogP`, `Molar.Refractivity`, `Fraction.sp3.Carbons`, `sp3.Carbons`, `Nitrogen.Count`, `Molecular.Weight`, `has_ester`, `has_carbonate`, `has_disulfide`

These are already computed by the LNPDB curators — we do not need to run RDKit ourselves. Removes a feature-engineering step from the plan.

## Target variable distribution by method

Experiment_value (overall): mean 0.92, std 12.3, min −10.7, max 222 — but heavily mixed-scale because the method varies. Per-method:

| Experiment_method | n | mean | std | min | max | scale |
|---|---:|---:|---:|---:|---:|---|
| `luminescence_normalized` | 14,302 | 0 | 1.00 | −3.16 | 10.27 | z-scored, primary target |
| `luminescence_discretized_normalized` | 3,058 | 0 | 1.00 | −2.19 | 5.47 | z-scored |
| `protein_abundance_normalized` | 766 | 0 | 1.00 | −1.90 | 6.04 | z-scored |
| `uptake` | 479 | 0 | 1.00 | −3.34 | 4.10 | z-scored |
| `editing_efficiency_normalized` | 141 | 0 | 1.00 | −1.16 | 5.57 | z-scored |
| `LRP6_knockdown_normalized` | 112 | 0 | 1.00 | −0.57 | 3.82 | z-scored |
| `luminescence_relative_to_Spikevax` | 60 | 0.44 | 0.43 | 0.003 | 1.54 | ratio |
| `diameter` | 96 | 167.9 | 26.5 | 109.2 | 222.1 | nm — physical, not transfection |
| `zeta_potential` | 96 | 2.34 | 5.65 | −10.69 | 14.51 | mV — physical |
| `hemolysis_percent` | 90 | 14.9 | 31.3 | 0.15 | 100 | % — toxicity |

**Decision for modeling (Day 7+):** primary target = `luminescence_normalized` (n=14,302 with non-null target). If sample size becomes a concern, combine with `luminescence_discretized_normalized` since both are z-scored and represent the same underlying biology.

## NA / missing patterns

- 269 rows have NaN across nearly every formulation column. These are physical-characterization-only rows (`diameter`, `zeta_potential`, `hemolysis_percent`) — keep for a separate physical-property analysis if interesting, but drop from the transfection-prediction model.
- 597 rows have NaN `Experiment_value` — drop for modeling.
- IL substructure columns (`IL_tail{2,3,4}_*`, `IL_linker_*`, `IL_head_*`) have varying NA depending on the lipid's structural complexity. Not used for the primary model since we have the global RDKit descriptors.

## Mol% sanity check

19,525 / 19,797 (98.6%) of rows have mol% components summing to exactly 100. The remaining 272 outliers (range: 0 to 125) should be dropped or flagged in cleaning.

## Ionizable lipid cardinality — modeling implication

13,028 unique ionizable lipids across 19,797 rows. Mean ≈ 1.5 LNPs per IL. Almost every formulation uses a unique IL. **One-hot encoding `IL_name` is not viable.** The IL must be represented by structural features — and conveniently the 17 pre-computed RDKit descriptors at the end of the schema give us exactly that. Helper / chol / PEG cardinality (7 / 16 / 15) is small enough to one-hot.

## Publications

42 unique PMIDs + 1 commercial supplier. Leave-one-publication-out (LOPO) cross-validation is the most honest generalization test — it asks "can the model predict efficiency for LNPs from a study it has never seen?"
