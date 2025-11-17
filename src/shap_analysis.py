import os
import argparse
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

def load_models(path):
    return joblib.load(path)

def compute_shap_for_model(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values

def save_global_summary(explainer, shap_values, X_sample, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=150)
    plt.clf()

def save_local_forceplots(explainer, shap_values, X, selected_indices, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    for i, idx in enumerate(selected_indices):
        row = X.iloc[idx:idx+1]
        sv = explainer.shap_values(row)
        # Save interactive HTML force plot
        force_html = shap.force_plot(explainer.expected_value, sv, row, matplotlib=False, show=False)
        shap.save_html(os.path.join(out_dir, f"force_plot_{i}_idx{idx}.html"), force_html)
        # Save static waterfall (bar-like) PNG via waterfall_legacy or waterfall
        try:
            shap.plots._waterfall.waterfall_legacy(explainer.expected_value, sv[0], feature_names=row.columns, max_display=10, show=False)
        except Exception:
            # fallback to simple bar plot
            contribs = pd.Series(sv[0], index=row.columns).sort_values(key=abs, ascending=False)[:10]
            contribs.plot.barh()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"force_plot_{i}_idx{idx}.png"), dpi=150, bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/german_credit.csv")
    parser.add_argument("--target", type=str, default="Risk")
    parser.add_argument("--models", type=str, default="artifacts/xgb_models_all_folds.pkl")
    parser.add_argument("--out_dir", type=str, default="artifacts")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df.copy()
    # Preprocessing must match what's done in training
    target = args.target
    if target not in df.columns:
        raise SystemExit(f"Target {target} not in data")
    # Simple preprocess: fill numeric NA with median, label-encode other cols as in train script
    for col in df.columns:
        if col == target:
            continue
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].astype(str).fillna("NA")
            df[col] = df[col].factorize()[0]

    X = df.drop(columns=[target])
    models = joblib.load(args.models)
    # Use the first model for SHAP TreeExplainer (or ensemble aggregation)
    model = models[0] if isinstance(models, list) else models
    # sample for SHAP speed
    X_sample = X.sample(n=min(2000, len(X)), random_state=42)
    explainer, shap_values = compute_shap_for_model(model, X_sample)
    save_global_summary(explainer, shap_values, X_sample, out_dir=args.out_dir)

    # Compute average predicted probability across all folds (if list of models)
    if isinstance(models, list):
        proba_all = np.mean([m.predict_proba(X)[:,1] for m in models], axis=0)
    else:
        proba_all = model.predict_proba(X)[:,1]

    idx_high = int(np.argmax(proba_all))
    idx_low = int(np.argmin(proba_all))
    idx_border = int(np.argmin(np.abs(proba_all - 0.5)))
    selected = [idx_high, idx_low, idx_border]
    save_local_forceplots(explainer, shap_values, X, selected, out_dir=args.out_dir)
    print(f"Saved SHAP outputs in {args.out_dir}")