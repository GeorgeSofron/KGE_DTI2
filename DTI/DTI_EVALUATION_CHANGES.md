# DTI Evaluation - Dimension Support Implementation

## Summary
Updated the DTI evaluation script to properly handle all embedding dimensions (100, 200, 300) and save results per dimension with a **Model → Dimension hierarchy**.

## Output Structure (Option B)

```
outputs_dti_evaluation_fixed/
├── TransE/                          # Model directory
│   ├── dim_100/
│   │   ├── transe_metrics.csv       # Results for TransE @ dim 100
│   │   ├── transe_evaluation_report.txt
│   │   └── figures/
│   │       ├── roc_curves.png
│   │       ├── pr_curves.png
│   │       └── auc_comparison.png
│   ├── dim_200/
│   │   └── [same structure]
│   ├── dim_300/
│   │   └── [same structure]
│   ├── transe_metrics_all_dims.csv  # Master for TransE (all dims)
│   └── transe_evaluation_report.txt # Master report for TransE
├── ComplEx/                         # Model directory
│   ├── dim_100/ [same structure]
│   ├── dim_200/ [same structure]
│   ├── dim_300/ [same structure]
│   ├── complex_metrics_all_dims.csv
│   └── complex_evaluation_report.txt
├── TriModel/                        # Model directory
│   ├── dim_100/ [same structure]
│   ├── dim_200/ [same structure]
│   ├── dim_300/ [same structure]
│   ├── trimodel_metrics_all_dims.csv
│   └── trimodel_evaluation_report.txt
├── dti_metrics_all_models_dims.csv  # Master results (all models + all dims)
└── dti_evaluation_master_report.txt # Master summary report
```

## Key Advantages of This Structure

### ✅ **Model-Focused Analysis**
- Easy to compare one model across dimensions
- Model-specific master CSV shows performance trajectory across dims
- Model-specific plots and reports for presentations

### ✅ **Individual Model Isolation**
- Each model gets its own directory tree
- Can delete/regenerate single model results without affecting others
- Clear separation of concerns

### ✅ **Scalable**
- Adding a new model? Create a new model directory
- Adding a new dimension? Just a new dim_* folder in each model
- Easy to parallelize per-model evaluation

## Changes Made

### 1. Reorganized Loop Structure

**Before:** Dimension loop (outer) → Model loop (inner)
**After:** Model loop (outer) → Dimension loop (inner)

```python
# New structure
for model_name in ['TransE', 'ComplEx', 'TriModel']:
    model_output_dir = f"{cfg.output_dir}/{model_name}"
    for dim in EMBEDDING_DIMS:
        dim_output_dir = f"{model_output_dir}/dim_{dim}"
```

### 2. Per-Dimension Files per Model

Each dimension-specific folder contains:
- `{model_name}_metrics.csv` - Single row with results for that dimension
- `{model_name}_evaluation_report.txt` - Detailed report for that dimension
- `figures/` - ROC, PR, and AUC comparison plots for that dimension

### 3. Per-Model Master Files

Each model directory contains:
- `{model_name}_metrics_all_dims.csv` - All dimensions for that model
- `{model_name}_evaluation_report.txt` - Summary across all dimensions

**Example:** `TransE/transe_metrics_all_dims.csv`
```
Model,Dimension,AUC-ROC,AUC-PR,Best_F1,...
TransE,100,0.7234,0.6890,0.6543,...
TransE,200,0.7345,0.7012,0.6654,...
TransE,300,0.7456,0.7123,0.6789,...
```

### 4. Global Master Files

In the root output directory:
- `dti_metrics_all_models_dims.csv` - All models and all dimensions
- `dti_evaluation_master_report.txt` - Summary for all combinations

## CSV Output Structure

### Dimension-Specific CSVs
```csv
Model,Dimension,AUC-ROC,AUC-PR,Best_F1,Best_Threshold,Precision@Best_F1,Recall@Best_F1,n_positives,n_negatives,dropped_test_positives,neg_ratio,neg_sampling,seed
TransE,100,0.7234,0.6890,0.6543,0.5234,0.6123,0.7012,452,4520,2,...
```

### Model Master CSVs
```csv
Model,Dimension,AUC-ROC,AUC-PR,Best_F1,Best_Threshold,Precision@Best_F1,Recall@Best_F1,n_positives,n_negatives,dropped_test_positives,neg_ratio,neg_sampling,seed
TransE,100,0.7234,0.6890,0.6543,0.5234,0.6123,0.7012,452,4520,2,...
TransE,200,0.7345,0.7012,0.6654,0.5123,0.6234,0.7123,452,4520,2,...
TransE,300,0.7456,0.7123,0.6789,0.5345,0.6456,0.7234,452,4520,2,...
```

### Global Master CSV
```csv
Model,Dimension,AUC-ROC,AUC-PR,Best_F1,Best_Threshold,Precision@Best_F1,Recall@Best_F1,n_positives,n_negatives,dropped_test_positives,neg_ratio,neg_sampling,seed
TransE,100,0.7234,0.6890,0.6543,0.5234,0.6123,0.7012,452,4520,2,...
TransE,200,0.7345,0.7012,0.6654,0.5123,0.6234,0.7123,452,4520,2,...
...
ComplEx,100,0.7456,0.7123,0.6789,0.5345,0.6456,0.7234,452,4520,2,...
...
```

## Enhanced Console Output

```
DTI EVALUATION (FIXED & MULTI-DIMENSION)
Dimensions to evaluate: [100, 200, 300]

======= EVALUATING MODEL: TransE =======
  ------
  Dimension: 100
  ------
    Loading TransE (dim=100)...
    ✅ Results:
       AUC-ROC: 0.7234
       AUC-PR : 0.6890
       Best F1: 0.6543  (thr=0.5234)
       Pos/Neg: 452/4520
    ✅ Saved: TransE/dim_100/transe_metrics.csv
    ✅ Saved: TransE/dim_100/transe_evaluation_report.txt

  ------
  Dimension: 200
  ------
    [Results for TransE @ dim 200]

  ✅ Saved model master results: TransE/transe_metrics_all_dims.csv
  ✅ Saved model report: TransE/transe_evaluation_report.txt

======= EVALUATING MODEL: ComplEx =======
  [Same structure for ComplEx]

======= EVALUATING MODEL: TriModel =======
  [Same structure for TriModel]

======= AGGREGATED RESULTS (ALL MODELS & DIMENSIONS) =======
Results by Model and Dimension:
Model     Dimension  AUC-ROC  AUC-PR  Best_F1
TransE    100          0.7234   0.6890    0.6543
TransE    200          0.7345   0.7012    0.6654
TransE    300          0.7456   0.7123    0.6789
ComplEx   100          0.7456   0.7123    0.6789
...
✅ Saved master results: dti_metrics_all_models_dims.csv
✅ DTI EVALUATION COMPLETE
```

## Key Features

### ✅ Fair Comparison
- Same negatives across all dimensions and models
- Same entity vocabulary
- Deterministic (fixed seed)

### ✅ Complete Organization
- Each model isolated in its own directory
- Each dimension clearly separated
- Multiple aggregation levels (per-dim, per-model, global)

### ✅ Easy Analysis
- To analyze TransE: `cat TransE/transe_metrics_all_dims.csv`
- To analyze dim_100 across all models: Filter `dti_metrics_all_models_dims.csv` by dimension
- To compare all models at dim_200: See dimension-specific results in each model's dim_200 folder

### ✅ Extensibility
- Add new dimension? Automatically handled
- Add new model? Create new model directory
- Rerun single model? Results isolated, no impact on others

## Running the Evaluation

```bash
cd DTI
python dti_evaluation.py
```

Expected runtime: ~5-10 minutes per dimension per model (depending on hardware)

## Output Inspection Examples

**View TransE results across all dimensions:**
```bash
cat TransE/transe_metrics_all_dims.csv
```

**View all models at dimension 100:**
```bash
cat TransE/dim_100/transe_metrics.csv
cat ComplEx/dim_100/complex_metrics.csv
cat TriModel/dim_100/trimodel_metrics.csv
```

**View global summary:**
```bash
cat dti_evaluation_master_report.txt
```

**Load and analyze in Python:**
```python
import pandas as pd

# Load all results
df = pd.read_csv('dti_metrics_all_models_dims.csv')

# Get TransE across all dims
print(df[df['Model'] == 'TransE'])

# Get all models at dim 200
print(df[df['Dimension'] == 200])

# Best model-dimension combo
print(df.loc[df['AUC-ROC'].idxmax()])
```

## File Organization Checklist

After running, verify these files exist:

```bash
# Model directories with dimension subdirectories
ls TransE/dim_*/transe_*.csv
ls ComplEx/dim_*/complex_*.csv  
ls TriModel/dim_*/trimodel_*.csv

# Model master files
ls TransE/transe_metrics_all_dims.csv
ls ComplEx/complex_metrics_all_dims.csv
ls TriModel/trimodel_metrics_all_dims.csv

# Global master files
ls dti_metrics_all_models_dims.csv
ls dti_evaluation_master_report.txt
```

## Future Enhancements

1. **Per-Model Parallelization** - Evaluate models in parallel
2. **Cross-Dimension Analysis** - Statistical tests comparing dimensions
3. **Model Comparison Plots** - Side-by-side model comparisons per dimension
4. **Performance Tracking** - Track improvements across model versions
5. **JSON Export** - Export results to JSON for external tools

