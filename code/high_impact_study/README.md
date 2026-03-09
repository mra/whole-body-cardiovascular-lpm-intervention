# High-Impact Intervention Study (Isolated Workspace)

This folder contains an isolated intervention-study workflow and manuscript copy edits,
so the current paper/code outside this folder are not disturbed.

## Study hypothesis
In hypotension, vasopressor-like vasoconstriction restores MAP but can reduce regional organ perfusion.

## Scenario design
- `S0`: baseline
- `S1`: hypotension (`LV.EA x0.80`, `RV.EA x0.85`, `C_SVC x1.20`, `C_IVC x1.20`)
- `S2`: S1 + low pressor (`R_CerT,R_RenT,R_MesT,R_HepT,R_SplT,R_ULimbT,R_LLimbT x1.10`)
- `S3`: S1 + medium pressor (same resistances `x1.25`)
- `S4`: S1 + high pressor (same resistances `x1.40`)

## Run
From repository root:

```bash
python -u code/high_impact_study/run_intervention_study.py \
  --n-cycles 101 \
  --tspan 800 \
  --n-samples 20 \
  --uq-n-cycles 41 \
  --uq-tspan 400 \
  --seed 7 \
  --output-dir code/high_impact_study/output_n20_uq41
```

To run full uncertainty at 100 samples:

```bash
python -u code/high_impact_study/run_intervention_study.py \
  --n-cycles 101 \
  --tspan 800 \
  --n-samples 100 \
  --seed 7 \
  --output-dir code/high_impact_study/output_n100
```

## Outputs
- `output_n20_uq41/figures/Figure_H1_intervention_dose_response.png`
- `output_n20_uq41/tables/deterministic_metrics_by_scenario.csv`
- `output_n20_uq41/tables/intervention_effect_sizes.csv`
- `output_n20_uq41/tables/uncertainty_summary.csv`
- `output_n20_uq41/scenario_parameters.json`

## tspan sensitivity check (for uncertainty runs)
Use this script to compare candidate final-cycle sampling resolutions (`tspan`) against a
high-resolution reference and recommend the smallest acceptable value.

```bash
python -u code/high_impact_study/check_tspan_sensitivity.py \
  --scenario S4 \
  --n-cycles 41 \
  --candidate-tspans 100,200,400,800 \
  --reference-tspan 1600 \
  --n-samples 5 \
  --tol-pct 0.5 \
  --output-dir code/high_impact_study/tspan_sensitivity_output
```

Outputs:
- `tspan_sensitivity_output/tspan_sensitivity_sample_level.csv`
- `tspan_sensitivity_output/tspan_sensitivity_summary.csv`
- `tspan_sensitivity_output/tspan_sensitivity_config.json`

## Paper copy edits
Edited copy files are in:
- `code/high_impact_study/paper_draft/sn_article.tex`
- `code/high_impact_study/paper_draft/supp_info.tex`
- `code/high_impact_study/paper_draft/supp_main.tex`

The new additions are in `paper_draft/sn_article.tex` only:
- Methods subsection: hypothesis-driven intervention study
- Results subsection: intervention outcomes with new figure/table
- Discussion paragraph: translational interpretation of pressure-perfusion trade-offs
