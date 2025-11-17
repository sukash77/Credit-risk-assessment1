# SHAP Explainability for Credit Risk (XGBoost)

This repository implements the project:
"Advanced Model Explainability: SHAP Analysis on a High-Dimensional Financial Prediction Task"

Contents:
- data/: place your dataset CSV here (see instructions below)
- src/train_xgboost.py — training pipeline (XGBoost) using StratifiedKFold, saves OOF predictions and models
- src/shap_analysis.py — compute SHAP (TreeExplainer) global summary and three local explanations (force plots)
- notebooks/colab_cells.txt — copy/paste cells into Google Colab (includes Kaggle download instructions)
- report/report.md — narrative report skeleton for interpretations and recommendations
- requirements.txt — Python deps
- .gitignore

Dataset
- You asked to use the Kaggle notebook link: https://www.kaggle.com/code/heidarmirhajisadati/safe-credit-risk-assessment-model-german
- That notebook uses a German credit dataset. Place the CSV at `data/german_credit.csv` (or update the path in scripts).
- Preferred Colab workflow (recommended): upload your `kaggle.json` (Kaggle API token) to Colab and download the dataset directly (cells provided in notebooks/colab_cells.txt). Alternatively, upload the dataset CSV to the repo `data/` folder.

How to run in Colab (short)
1. Open a new Colab notebook and paste the cells from `notebooks/colab_cells.txt` in order.
2. Upload `kaggle.json` when requested (or upload the dataset CSV directly to /content/data/).
3. Run EDA cell, training cell, and SHAP cell. Outputs will be saved to `/content/drive/MyDrive/shap-credit-xgboost/` if you mount Drive.

How to create the GitHub repo locally and push (one-time)
1. Create a new repository on GitHub named `shap-credit-xgboost` (or use your existing repo).
2. On your machine:
   git init
   git add .
   git commit -m "Initial commit: SHAP + XGBoost credit risk project"
   git branch -M main
   git remote add origin https://github.com/<your-username>/shap-credit-xgboost.git
   git push -u origin main

If you want me to push to your account, create the empty repo and tell me its exact name & owner; I will then request the push authorization (you will see an approval dialog — accept it) and I will push the prepared files.

Notes
- I used XGBoost per your request and classification task.
- The scripts use simple preprocessing (median fill for numeric). If the Kaggle dataset has categorical features, the Colab notebook includes encoding suggestions.
- After you run the notebook or scripts, send me the OOF AUC and the saved SHAP plots and I will produce the filled report and three actionable recommendations derived only from SHAP.
