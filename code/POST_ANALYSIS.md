# Post-Analysis Plot Pipeline

Run from repo root:

```bash
python code/post_analysis.py
```

To regenerate Sobol sensitivity figures:

```bash
python code/post_analysis.py --run-sobol --sobol-samples 64 --sobol-n-cycles 40 --sobol-tspan 400
```

## What it generates

Main manuscript figures:
- `code/output/figures/main/Figure4.png`
- `code/output/figures/main/Figure5.png`
- `code/output/figures/main/Figure7.png`
- `code/output/figures/main/Figure6.png` (with `--run-sobol`)

Supplementary figures:
- `code/output/figures/supp/pulmonary_waveform.png`
- `code/output/figures/supp/major_arteries_waveform.png`
- `code/output/figures/supp/upperbody_waveform.png`
- `code/output/figures/supp/lowerbody_waveform.png`
- `code/output/figures/supp/Organ_flows.png`
- `code/output/figures/supp/vein_flows.png`
- `code/output/figures/supp/Fig8.png`
- `code/output/figures/supp/Fig9.png`
- `code/output/figures/supp/Fig11.png`
- `code/output/figures/supp/dumbbell_plot_physical_vals.png`
- `code/output/figures/supp/Fig7.png` (with `--run-sobol`)
- `code/output/figures/supp/Top_5_ST_Sobol_Indices_base.png` (with `--run-sobol`)
- `code/output/figures/supp/global_ST_basemodel_heatmap.png` (with `--run-sobol`)

Analysis tables:
- `code/output/analysis/min_max_values.csv`
- `code/output/analysis/mean_values.csv`
- `code/output/analysis/sobol_st_indices.csv` (with `--run-sobol`)
- `code/output/analysis/sobol_st_confidence.csv` (with `--run-sobol`)
- `code/output/analysis/sobol_metadata.json` (with `--run-sobol`)

## Notes

- Core model scripts are preserved: `code/dynamics.py`, `code/data_structures.py`, `code/utils.py`.
- Notebook files were removed.
- Sobol figures are computed directly from model re-simulations using SALib (not from precomputed index tables).
