"""
Baseline text model: extract simple features from catalog_content and train a regression (RandomForest).
Usage:
 python src/train_text_model.py --csv data/reduced_train.csv --model-out model/text_model.joblib
"""
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from src.utils import setup_logging, makedirs
import logging

setup_logging()
logger = logging.getLogger(__name__)

def train(csv_path, model_out):
    df = pd.read_csv(csv_path)
    X = df["catalog_content"].fillna("")
    y = df["price"].astype(float)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("rf", RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    logger.info("Train R2: %.4f", pipe.score(X_train, y_train))
    logger.info("Val R2: %.4f", pipe.score(X_val, y_val))
    makedirs(Path(model_out).parent)
    joblib.dump(pipe, model_out)
    logger.info("Saved text model to %s", model_out)

if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model-out", required=True)
    args = parser.parse_args()
    train(args.csv, args.model_out)