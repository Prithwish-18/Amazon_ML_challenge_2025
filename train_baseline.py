"""
train_baseline.py
- Loads X_text (sparse .npz) and y (npy)
- Trains LightGBM regression baseline
- Saves model and metrics
"""
import argparse
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import sparse
from joblib import dump
import json
from src.utils import ensure_dir, logger

def main(args):
    X = sparse.load_npz(args.features)
    y = np.load(args.targets)
    logger.info("Features shape: %s, targets shape: %s", X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": args.random_state,
        "learning_rate": 0.1,
        "num_leaves": 31
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        early_stopping_rounds=50,
        verbose_eval=50
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    mae = mean_absolute_error(y_val, val_pred)
    metrics = {"rmse": float(rmse), "mae": float(mae)}
    logger.info("Validation metrics: %s", metrics)

    ensure_dir(args.out_dir)
    dump(model, f"{args.out_dir}/lgbm_model.joblib")
    with open(f"{args.out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to X_text.npz")
    parser.add_argument("--targets", required=True, help="Path to y.npy")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    main(args)