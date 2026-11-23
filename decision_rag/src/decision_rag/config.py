"""Config and paths. Small, committed outputs live in the repo; large data lives under $DRAG_DIR."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # decision_rag/


def load_config(path=None):
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def data_dir():
    return Path(os.environ.get("DRAG_DIR", Path.home() / "scratch" / "decision_rag"))


def nq_dir():
    return data_dir() / "data" / "nq"


def tree_dir(seed):
    return data_dir() / "trees" / f"nq_seed{seed}"


def splits_dir():
    return Path(os.environ.get("DRAG_SPLITS_DIR", ROOT / "splits"))
