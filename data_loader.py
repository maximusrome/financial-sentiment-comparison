"""
Load Financial PhraseBank (Malo et al. 2014) and build the shared
train/val/test split used by every model in the project.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
SPLIT_PATH = DATA_DIR / "splits.parquet"
METADATA_PATH = REPO_ROOT / "predictions" / "test_metadata.csv"


def _load_raw_from_hf() -> pd.DataFrame:
    """Download the FPB zip from HuggingFace and parse the four
    agreement-tier text files. The dataset nests: every sentence in
    'allagree' also appears in 75/66/50agree, so we assign each sentence
    to its highest tier."""

    url = ("https://huggingface.co/datasets/takala/financial_phrasebank/"
           "resolve/main/data/FinancialPhraseBank-v1.0.zip")
    print(f"Downloading {url} ...")

    # Browser UA because HF sometimes rate-limits plain Python requests
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        zip_bytes = resp.read()
    print(f"  Downloaded {len(zip_bytes) / 1024:.0f} KB")

    tier_files = {
        "50":  "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
        "66":  "FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
        "75":  "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
        "all": "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
    }

    def parse(content: bytes) -> pd.DataFrame:
        # latin-1 because the corpus uses Windows-1252 chars (e.g. long dash)
        text = content.decode("latin-1")
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Format: <sentence>@<label>. rpartition handles '@' in sentences.
            sent, sep, label = line.rpartition("@")
            if not sep:
                continue
            rows.append({"sentence": sent.strip(),
                         "label_name": label.strip().lower()})
        return pd.DataFrame(rows)

    by_config: Dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for tier_key, fname in tier_files.items():
            with zf.open(fname) as f:
                by_config[tier_key] = parse(f.read())

    # 50agree has every sentence; use as base and tag non-overlapping tiers
    full = by_config["50"].drop_duplicates(subset=["sentence"]).copy()
    all_agree = set(by_config["all"]["sentence"])
    agree_75  = set(by_config["75"]["sentence"])
    agree_66  = set(by_config["66"]["sentence"])

    def tier(sent: str) -> str:
        if sent in all_agree: return "100"
        if sent in agree_75:  return "75-99"
        if sent in agree_66:  return "66-74"
        return "50-65"

    full["agreement_tier"] = full["sentence"].map(tier)
    full["label"] = full["label_name"].map(LABEL2ID).astype(int)
    full = full.rename(columns={"sentence": "text"})
    full["sentence_id"] = [f"s{i:05d}" for i in range(len(full))]
    return full[["sentence_id", "text", "label", "agreement_tier"]].reset_index(drop=True)


def _stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """80/10/10 split stratified by (label x agreement_tier) so every tier
    and class is represented in each subset."""
    df = df.copy()
    df["_strat"] = df["label"].astype(str) + "_" + df["agreement_tier"]

    trainval_idx, test_idx = train_test_split(
        df.index, test_size=TEST_FRAC,
        stratify=df["_strat"], random_state=RANDOM_SEED,
    )
    # After pulling out test, val takes 1/9 of the remaining 90% = 10% overall
    val_size = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_size,
        stratify=df.loc[trainval_idx, "_strat"], random_state=RANDOM_SEED,
    )

    df["split"] = "train"
    df.loc[val_idx, "split"]  = "val"
    df.loc[test_idx, "split"] = "test"
    return df.drop(columns=["_strat"])


def build_and_save_split(force: bool = False) -> pd.DataFrame:
    """Build the canonical split and save it. Only runs once per project -
    the saved file is committed to git so every teammate gets identical data."""
    if SPLIT_PATH.exists() and not force:
        print(f"Split already exists at {SPLIT_PATH}. Pass force=True to regenerate.")
        return pd.read_parquet(SPLIT_PATH)

    print("Loading Financial PhraseBank from HuggingFace...")
    df = _load_raw_from_hf()
    print(f"  Loaded {len(df)} unique sentences.")
    print("  Agreement tier distribution:")
    print(df["agreement_tier"].value_counts().sort_index().to_string())
    print("  Label distribution:")
    print(df["label"].map(ID2LABEL).value_counts().to_string())

    print("\nCreating stratified 80/10/10 split...")
    df = _stratified_split(df)
    print(df["split"].value_counts().to_string())

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SPLIT_PATH, index=False)
    print(f"\nSaved split to {SPLIT_PATH}")

    # Test metadata (sentence_id -> true label) is used by the evaluation
    # framework to score each teammate's predictions CSV.
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_md = df[df["split"] == "test"][
        ["sentence_id", "text", "label", "agreement_tier"]
    ].copy()
    test_md["label_name"] = test_md["label"].map(ID2LABEL)
    test_md.to_csv(METADATA_PATH, index=False)
    print(f"Saved test metadata to {METADATA_PATH}")
    return df


def load_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (df_train, df_val, df_test)."""
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"No split at {SPLIT_PATH}. Run `python data_loader.py` first."
        )
    df = pd.read_parquet(SPLIT_PATH)
    cols = ["sentence_id", "text", "label", "agreement_tier"]
    return (df[df["split"] == "train"][cols].reset_index(drop=True),
            df[df["split"] == "val"][cols].reset_index(drop=True),
            df[df["split"] == "test"][cols].reset_index(drop=True))


def summarize_split() -> None:
    """Print split sizes and per-split label/tier distributions."""
    df_train, df_val, df_test = load_split()
    print(f"Train: {len(df_train)}   Val: {len(df_val)}   Test: {len(df_test)}")
    print("\nLabel distribution by split:")
    for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"  {name:6s}: {d['label'].map(ID2LABEL).value_counts().to_dict()}")
    print("\nAgreement tier distribution by split:")
    for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"  {name:6s}: {d['agreement_tier'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    build_and_save_split()
    print()
    summarize_split()
