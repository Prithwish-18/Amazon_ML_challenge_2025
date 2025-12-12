"""
Download images from CSV image_link column.
Usage:
  python src/download_images.py --csv data/reduced_train.csv --out-dir data/images --id-col sample_id --url-col image_link
"""
import argparse
import os
import requests
from pathlib import Path
import logging
from tqdm import tqdm
import pandas as pd
from src.utils import makedirs, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def download_image(url, path, timeout=10):
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            with open(path, "wb") as f:
                f.write(resp.content)
            return True
        else:
            logger.warning("Bad status %s for %s", resp.status_code, url)
    except Exception as e:
        logger.warning("Error %s for %s", e, url)
    return False

def main(csv_path, out_dir, id_col="sample_id", url_col="image_link", limit=None):
    df = pd.read_csv(csv_path)
    makedirs(out_dir)
    rows = df.to_dict(orient="records")
    if limit:
        rows = rows[:limit]
    for r in tqdm(rows, total=len(rows)):
        sid = r[id_col]
        url = r[url_col]
        ext = Path(url).suffix.split("?")[0] or ".jpg"
        path = Path(out_dir) / f"{sid}{ext}"
        if path.exists():
            continue
        ok = download_image(url, path)
        if not ok:
            # try .jpg fallback
            download_image(url, Path(out_dir) / f"{sid}.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--url-col", default="image_link")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.csv, args.out_dir, args.id_col, args.url_col, args.limit)