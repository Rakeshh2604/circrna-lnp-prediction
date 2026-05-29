# circRNA-cut summary

## Subsets and prediction envelopes

| Subset | n | median predicted | 90th percentile predicted |
|---|---:|---:|---:|
| All LNPs | 14,302 | -0.037 | +0.656 |
| mRNA | 11,892 | -0.012 | +0.646 |
| circRNA-like (mRNA, IL:NA mass ratio > 10) | 2,096 | +0.079 | +0.895 |

## Top 5 features by mean |SHAP|, per subset


**All LNPs (n=14,302)**

| Feature | mean \|SHAP\| |
|---|---:|
| `rotatable_bonds` | 0.1108 |
| `topological_polar_surface_area` | 0.0737 |
| `peg_molratio` | 0.0674 |
| `nitrogen_count` | 0.0653 |
| `logp` | 0.0651 |

**mRNA (n=11,892)**

| Feature | mean \|SHAP\| |
|---|---:|
| `rotatable_bonds` | 0.1135 |
| `topological_polar_surface_area` | 0.0736 |
| `logp` | 0.0669 |
| `nitrogen_count` | 0.0594 |
| `peg_molratio` | 0.0593 |

**circRNA-like (mRNA, IL:NA mass ratio > 10) (n=2,096)**

| Feature | mean \|SHAP\| |
|---|---:|
| `peg_molratio` | 0.1226 |
| `il_to_nucleicacid_massratio` | 0.1101 |
| `rotatable_bonds` | 0.1007 |
| `topological_polar_surface_area` | 0.0840 |
| `hl_molratio` | 0.0833 |

## Reading these numbers

If the top features in the circRNA-like subset look very similar to the overall top features, the model is saying the same chemistry levers matter even in the demanding-delivery regime — and those features become defensible hypotheses for circRNA delivery experiments (subject to the LOPO caveat that cross-study transfer is the unsolved problem).