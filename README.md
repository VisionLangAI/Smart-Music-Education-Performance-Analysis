# Smart Music Education Using Machine Learning and Deep Learning

This repository provides a **reproducible ML/DL pipeline** for predicting student performance in online music education, comparing multiple baselines with a **TabNet** model and validating generalization on an independent dataset.

## 1) What’s included

- **Dataset 1 (Kaggle)**: primary training + CV evaluation  
- **Dataset 2**: independent external validation  
- Preprocessing: imputation + scaling + categorical encoding  
- Feature Engineering: `Effort_Index`, `Focus_Ratio`, `Tempo_Rhythm_Ratio`, `Musical_Cluster`  
- Models:
  - Baselines: RandomForest, Ridge (and XGBoost if installed)
  - Proposed: TabNetRegressor (attention-based feature selection)
- Metrics: **MAE, RMSE, R², MAPE, SMAPE**
- External validation: train on Dataset 1 → test on Dataset 2

> Notes:
> - For tabular per-session data, MLP is a typical DL baseline. Sequence models (RNN/LSTM/BiLSTM) require **true time sequences**; add them only if your dataset contains session-wise sequences.

---

## 2) Requirements

Tested with **Python 3.8+**

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
pip install xgboost
pip install pytorch-tabnet
```

If you plan to add deep models (MLP/LSTM) using PyTorch:

```bash
pip install torch torchvision torchaudio
```

---

## 3) Data placement

Create:

```
data/
  dataset1.csv
  dataset2.csv
```

Your target column (default) is:

- `Performance_Score`

If your target has a different name, pass `--target YOUR_COLUMN`.

---

## 4) Run

```bash
python smart_music_education_pipeline.py \
  --data1 data/dataset1.csv \
  --data2 data/dataset2.csv \
  --target Performance_Score \
  --folds 5 \
  --out outputs
```

Outputs saved to:

- `outputs/cv_summary.csv`  (mean ± std across folds)
- `outputs/external_validation_tabnet.json` (TabNet on Dataset 2)

---

## 5) How stratified CV works (regression)

Regression targets don’t directly support StratifiedKFold.  
This pipeline **bins the target** into quantiles (default 5) to stratify folds and keep performance distribution balanced across splits.

---

## 6) Feature Engineering (implemented)

The script adds these features if the required columns exist:

- **Effort_Index** = `Mean_Energy_dB * Duration / (Tempo + 1)`
- **Focus_Ratio** = `Active_Time / Total_Time`
- **Tempo_Rhythm_Ratio** = `Tempo / (Rhythm + 1)`
- **Musical_Cluster**:
  - KMeans on spectral columns containing `Spectral`, `Pitch`, or `MFCC`
  - fallback to KMeans on `Tempo + Mean_Energy_dB`

If your dataset uses different column names, either:
1) rename your CSV columns to match, or  
2) edit `create_engineered_features()` inside the script.

---

## 7) Explainability (SHAP / LIME)

This pipeline focuses on training + evaluation.  
To add SHAP/LIME:
- store the fitted preprocessor and final model,
- export feature names (`preprocessor.get_feature_names_out()`),
- run SHAP KernelExplainer / LIME TabularExplainer on the transformed feature matrix.

---

## 8) Reproducibility & fairness (recommended reporting)

For reporting:
- state your split protocol: StratifiedKFold on binned target
- report mean ± std across folds
- include external validation results (Dataset 2)
- fairness analysis: compare residuals/errors across **age, gender, class level**

---

## 9) Files

- `smart_music_education_pipeline.py` — main runnable pipeline (CV + TabNet + external validation)
- `README.md` — this documentation
