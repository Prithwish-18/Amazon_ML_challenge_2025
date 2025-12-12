"""
Read CSV, reduce dataset (optional), save reduced CSV.
Usage:
  python src/data_prep.py --input data/train.csv --out-csv data/reduced_train.csv --sample 20000
"""
import argparse
import pandas as pd
from src.utils import makedirs, setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

def reduce_csv(input_csv, out_csv, n_sample=None, random_state=42):
    df = pd.read_csv(input_csv)
    logger.info("Loaded %d rows", len(df))
    if n_sample is not None and n_sample < len(df):
        df = df.sample(n=n_sample, random_state=random_state)
        logger.info("Sampled down to %d rows", len(df))
    makedirs(Path(out_csv).parent)
    df.to_csv(out_csv, index=False)
    logger.info("Saved reduced csv to %s", out_csv)

if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    reduce_csv(args.input, args.out_csv, args.sample)