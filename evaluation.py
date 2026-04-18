"""
Metrics and cross-model comparison. Reads each model's predictions CSV
from predictions/ and produces accuracy, macro-F1, per-class scores,
confusion matrices, per-tier breakdowns, and error examples.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import ID2LABEL, LABEL2ID, METADATA_PATH


LABEL_ORDER = ["negative", "neutral", "positive"]
TIER_ORDER = ["100", "75-99", "66-74", "50-65"]

# Expected filenames (falls back to file stem if a teammate uses a different name)
CANONICAL_MODELS = [
    "tfidf_logreg",          # Gavin
    "tfidf_logreg_lm",       # Noah
    "bert_finetuned",        # Avi
    "finbert_finetuned",     # Max
]


def _load_metadata() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"No test metadata at {METADATA_PATH}. "
            f"Run `python data_loader.py` first."
        )
    return pd.read_csv(METADATA_PATH)


def _load_predictions_file(path: str | Path) -> pd.DataFrame:
    """Load a predictions CSV. Labels may be strings or 0/1/2 ints."""
    df = pd.read_csv(path)
    missing = {"sentence_id", "predicted_label"} - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file {path} missing columns: {missing}")

    if df["predicted_label"].dtype.kind in "iu":
        df["predicted_label"] = df["predicted_label"].map(ID2LABEL)
    else:
        df["predicted_label"] = df["predicted_label"].str.strip().str.lower()

    unknown = set(df["predicted_label"]) - set(LABEL_ORDER)
    if unknown:
        raise ValueError(f"Unknown predicted labels in {path}: {unknown}")
    return df


def evaluate_predictions(predictions_path: str | Path,
                         metadata: Optional[pd.DataFrame] = None) -> Dict:
    """Compute overall and per-tier metrics for one model's predictions."""
    preds = _load_predictions_file(predictions_path)
    if metadata is None:
        metadata = _load_metadata()

    merged = metadata.merge(preds, on="sentence_id", how="inner",
                            validate="one_to_one")
    if len(merged) != len(metadata):
        missing = set(metadata["sentence_id"]) - set(preds["sentence_id"])
        raise ValueError(
            f"Predictions file is missing {len(missing)} test examples. "
            f"Every test sentence_id must have a prediction. "
            f"Example missing: {list(missing)[:3]}"
        )

    y_true = merged["label_name"].values
    y_pred = merged["predicted_label"].values

    def metrics_block(yt, yp):
        prec, rec, f1, support = precision_recall_fscore_support(
            yt, yp, labels=LABEL_ORDER, zero_division=0
        )
        return {
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, labels=LABEL_ORDER,
                                       average="macro", zero_division=0)),
            "per_class": {
                label: {"precision": float(prec[i]), "recall": float(rec[i]),
                        "f1": float(f1[i]), "support": int(support[i])}
                for i, label in enumerate(LABEL_ORDER)
            },
            "confusion_matrix": confusion_matrix(yt, yp, labels=LABEL_ORDER).tolist(),
            "n": int(len(yt)),
        }

    out = {
        "overall": metrics_block(y_true, y_pred),
        "by_tier": {},
        "n_test": int(len(merged)),
    }
    for tier in TIER_ORDER:
        mask = merged["agreement_tier"].astype(str) == tier
        if mask.sum() > 0:
            out["by_tier"][tier] = metrics_block(y_true[mask.values], y_pred[mask.values])
    return out


def load_all_predictions(predictions_dir: str | Path = "predictions"
                         ) -> Dict[str, pd.DataFrame]:
    """Load every *_predictions.csv in the predictions directory."""
    files = sorted(glob.glob(str(Path(predictions_dir) / "*_predictions.csv")))
    return {Path(f).stem.replace("_predictions", ""): _load_predictions_file(f)
            for f in files}


def build_comparison_table(predictions: Optional[Dict[str, pd.DataFrame]] = None,
                            predictions_dir: str | Path = "predictions"
                            ) -> pd.DataFrame:
    """Master results table: rows are (model, tier), columns are accuracy and
    macro_f1. Tier 'overall' aggregates across all test examples."""
    if predictions is None:
        predictions = load_all_predictions(predictions_dir)
    metadata = _load_metadata()

    rows = []
    for model in predictions:
        m = evaluate_predictions(
            Path(predictions_dir) / f"{model}_predictions.csv", metadata=metadata
        )
        rows.append({"model": model, "tier": "overall",
                     "accuracy": m["overall"]["accuracy"],
                     "macro_f1": m["overall"]["macro_f1"],
                     "n": m["overall"]["n"]})
        for tier, tm in m["by_tier"].items():
            rows.append({"model": model, "tier": tier,
                         "accuracy": tm["accuracy"], "macro_f1": tm["macro_f1"],
                         "n": tm["n"]})

    df = pd.DataFrame(rows)
    tier_cat = pd.CategoricalDtype(["overall"] + TIER_ORDER, ordered=True)
    df["tier"] = df["tier"].astype(tier_cat)
    model_cat = pd.CategoricalDtype(
        [m for m in CANONICAL_MODELS if m in set(df["model"])] +
        [m for m in df["model"].unique() if m not in CANONICAL_MODELS],
        ordered=True,
    )
    df["model"] = df["model"].astype(model_cat)
    return df.sort_values(["model", "tier"]).reset_index(drop=True)


def pretty_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Wide-form: rows are models, columns are (metric, tier)."""
    return df.pivot(index="model", columns="tier",
                    values=["accuracy", "macro_f1"]).round(4)


def find_misclassifications(predictions_path: str | Path,
                            n_examples: int = 10,
                            metadata: Optional[pd.DataFrame] = None
                            ) -> pd.DataFrame:
    """Return misclassified test examples, spread across (true_label, tier)
    cells where possible."""
    if metadata is None:
        metadata = _load_metadata()
    preds = _load_predictions_file(predictions_path)
    merged = metadata.merge(preds, on="sentence_id", how="inner")
    errors = merged[merged["label_name"] != merged["predicted_label"]].copy()
    if len(errors) <= n_examples:
        return errors.reset_index(drop=True)

    strat = errors["label_name"].astype(str) + "_" + errors["agreement_tier"].astype(str)
    per_cell = max(1, n_examples // strat.nunique())
    parts = [grp.sample(min(len(grp), per_cell), random_state=0)
             for _, grp in errors.groupby(strat, observed=True)]
    return pd.concat(parts).head(n_examples).reset_index(drop=True)


def plot_confusion_matrix(cm, labels: List[str] = LABEL_ORDER,
                          title: str = "", ax=None, normalize: bool = True):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        # Row-normalize so diagonal shows recall per class
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm / row_sums
        fmt = ".2f"
    else:
        cm_display, fmt = cm, ".0f"

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar=False, ax=ax, square=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    return ax


def plot_agreement_tier_comparison(comparison_df: pd.DataFrame,
                                   metric: str = "macro_f1",
                                   figsize=(8, 4)):
    """Grouped bar chart: x = agreement tier, bars = models, y = metric."""
    d = comparison_df[comparison_df["tier"] != "overall"].copy()
    pivot = d.pivot(index="tier", columns="model", values=metric).loc[TIER_ORDER]

    ax = pivot.plot(kind="bar", figsize=figsize, width=0.8)
    ax.set_xlabel("Annotator agreement tier")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by agreement tier")
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    return ax


def main():
    """CLI: print the comparison table and save it to disk."""
    preds = load_all_predictions("predictions")
    if not preds:
        print("No predictions found in predictions/. "
              "Add *_predictions.csv files and re-run.")
        return

    print(f"Found {len(preds)} models: {list(preds.keys())}\n")
    table = build_comparison_table(preds)
    print(pretty_comparison_table(table).to_string())

    out_path = Path("results/tables/comparison_table.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"\nSaved table to {out_path}")


if __name__ == "__main__":
    main()
