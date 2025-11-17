import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import xgboost as xgb

def basic_preprocess(df, target_col):
    # Simple preprocessing: fill numeric NAs with median, label-encode categorical
    df = df.copy()
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].astype(str).fillna("NA")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    return df

def train_xgboost(df, target_col='target', n_splits=5, random_state=42, out_dir='artifacts'):
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()
    df = basic_preprocess(df, target_col)
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(df))
    models = []

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 10000,
        "use_label_encoder": False,
        "verbosity": 1,
        "tree_method": "hist"
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=100)
        preds = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = preds
        models.append(model)
        joblib.dump(model, os.path.join(out_dir, f"xgb_model_fold{fold}.pkl"))
    auc = roc_auc_score(y, oof)
    pd.DataFrame({"oof": oof, target_col: y}).to_csv(os.path.join(out_dir, "oof_preds.csv"), index=False)
    joblib.dump(models, os.path.join(out_dir, "xgb_models_all_folds.pkl"))
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"OOF AUC: {auc:.6f}\n")
    print(f"OOF AUC: {auc:.6f}")
    return {"models": models, "oof": oof, "auc": auc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/german_credit.csv", help="path to dataset CSV")
    parser.add_argument("--target", type=str, default="Risk", help="target column name (default 'Risk')")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="artifacts")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    # If target column is not 0/1, map or convert it
    if args.target not in df.columns:
        raise SystemExit(f"Target column {args.target} not found in data columns: {df.columns.tolist()}")
    # Convert common label names to 0/1 if necessary
    # Example: if Risk = 'good'/'bad' or '1'/'2' adjust accordingly (user can edit mapping)
    # User can update this mapping according to the dataset used.
    try:
        df[args.target] = df[args.target].astype(int)
    except Exception:
        # Attempt mapping common labels
        df[args.target] = df[args.target].map(lambda x: 1 if str(x).lower() in ['bad', '1', 'yes', 'true'] else 0)

    train_xgboost(df, target_col=args.target, n_splits=args.n_splits, out_dir=args.out_dir)