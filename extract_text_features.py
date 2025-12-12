"""
extract_text_features.py
- Read CSV, vectorize `text_col` with TF-IDF
- Save sparse matrix (npz), ids and targets
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from joblib import dump
from src.utils import ensure_dir, logger

def main(args):
    df = pd.read_csv(args.input)
    logger.info("Read %d rows", len(df))

    texts = df[args.text_col].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)
    ensure_dir(args.out_dir)
    sparse.save_npz(os.path.join(args.out_dir, "X_text.npz"), X, compressed=True)
    np.save(os.path.join(args.out_dir, "ids.npy"), df[args.id_col].values)
    if args.target_col and args.target_col in df.columns:
        np.save(os.path.join(args.out_dir, "y.npy"), df[args.target_col].values)
    dump(vectorizer, os.path.join(args.out_dir, "vectorizer.joblib"))
    logger.info("Saved X_text.npz, ids.npy, (y.npy if target present) and vectorizer.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-col", default="catalog_content")
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--target-col", default="price")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-features", type=int, default=10000)
    args = parser.parse_args()
    main(args)