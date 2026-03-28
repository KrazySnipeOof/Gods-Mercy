# Gods-Mercy
Team God's Mercy - ARD 2026 Maize Hackathon: AI/ML genomic prediction of maize agronomic performance (yield, plant height, moisture) using GBLUP/ridge regression on SNP markers with multi-environment data (2001-2008).

## Environment-Specific Decision Support (Option B)

This repository now includes an environment-specific maize yield pipeline:
- predicts yield per `line x location x year`,
- combines genetic markers, environmental covariates, and historical phenotype traits,
- captures genotype-by-environment (GxE) effects using explicit interaction features,
- compares a baseline model vs a stronger final model under grouped CV by environment,
- ranks top candidate lines per environment for advancement decisions.

Decision-support framing:
"A model that predicts maize yield using genetic and environmental data to help select the best lines under limited resources."

Why environment-specific modeling is used here:
- It is closer to real breeding decisions (selection is made per target environment).
- It captures local adaptation.
- It improves targeted recommendations for each location-year.
- It better handles genotype-by-environment effects.

## Run

Install dependencies:

```powershell
python -m pip install pandas numpy scikit-learn
```

Run the pipeline:

```powershell
python maize_yield_decision_support.py `
  --environment-path "C:\path\to\environment.csv" `
  --genomic-path "C:\path\to\genomic.csv" `
  --target-col YLD_BE `
  --output-dir outputs
```

Notes:
- `--phenotype-path` is now optional for first full train runs.
- If omitted, the script auto-selects cohort-matched phenotype:
  - C1 genotypes -> `C1_Phenotype_Data_V2.csv`
  - C2 genotypes -> `C2_Phenotype_Data_V2.csv`

## Outputs

The script writes:
- `outputs/prediction_results.csv`
- `outputs/ranked_candidates_per_environment.csv`
- `outputs/feature_importance_model_gain.csv`
- `outputs/feature_importance_permutation.csv`
- `outputs/environment_level_metrics.csv`
- `outputs/baseline_fold_metrics.csv`
- `outputs/final_fold_metrics.csv`
- `outputs/run_summary.json`
