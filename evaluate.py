"""
evaluate.py
- Load model and features, produce predictions CSV
"""
import argparse
import numpy as np
from scipy import sparse
from joblib import load
import pandas as pd
from src.utils import ensure_dir, logger

def main(args):
    X = sparse.load_npz(args.features)
    ids = np.load(args.ids) if args.ids else None
    logger.info("Loaded features shape: %s", X.shape)

    model = load(args.model)
    preds = model.predict(X)

    out = pd.DataFrame({
        "sample_id": ids if ids is not None else np.arange(len(preds)),
        "pred_price": preds
    })
    ensure_dir(os.path.dirname(args.out) or ".")
    out.to_csv(args.out, index=False)
    logger.info("Wrote predictions to %s", args.out)

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--targets", help="Optional y.npy (for metrics)", default=None)
    parser.add_argument("--ids", default="./artifacts/ids.npy")
    parser.add_argument("--out", default="./artifacts/preds.csv")
    args = parser.parse_args()
    main(args)