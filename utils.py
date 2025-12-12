import os
import logging
from pathlib import Path

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=level
    )

def makedirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)