"""
Fine-tune FinBERT (or bert-base-uncased) on Financial PhraseBank and
run a small learning-rate x epochs grid search.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from sklearn.metrics import accuracy_score, f1_score

from data_loader import ID2LABEL, LABEL2ID, load_split


@dataclass
class TrainConfig:
    model_name: str = "ProsusAI/finbert"
    output_name: str = "finbert_finetuned"
    max_length: int = 128                 # FPB sentences are short
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    predictions_path: str = "predictions/finbert_finetuned_predictions.csv"
    metrics_path: str = "results/tables/finbert_finetuned_metrics.json"
    model_save_dir: Optional[str] = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism trade-off: slower but reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PhraseBankDataset(Dataset):
    """Tokenizes sentences on demand so we don't hold 4K padded tensors in RAM."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate_loader(model, loader, device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            all_preds.append(out.logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(batch["labels"].numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro",
                        labels=[0, 1, 2], zero_division=0))
    return acc, f1, y_true, y_pred


def train_one_config(cfg: TrainConfig,
                     df_train: Optional[pd.DataFrame] = None,
                     df_val: Optional[pd.DataFrame] = None,
                     df_test: Optional[pd.DataFrame] = None,
                     verbose: bool = True) -> Dict:
    """Fine-tune one config, evaluate on val and test, save predictions."""
    seed_everything(cfg.seed)
    device = _get_device()
    if verbose:
        print(f"Device: {device}")
        print(f"Config: {cfg}")

    if df_train is None or df_val is None or df_test is None:
        df_train, df_val, df_test = load_split()

    # Load encoder and reinitialize the classifier head.
    # ProsusAI/finbert ships with a head already fine-tuned on Financial
    # PhraseBank (Araci 2019). Keeping it would give FinBERT a warm start on
    # our exact task while BERT (no pretrained head) gets a random init,
    # confounding RQ1. RQ1 is about the encoder, so we reset the head to a
    # fresh random Linear with BERT's init scheme.
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=3, ignore_mismatched_sizes=True,
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    model.classifier = nn.Linear(model.config.hidden_size, 3)
    model.classifier.apply(model._init_weights)
    model = model.to(device)

    train_ds = PhraseBankDataset(df_train, tokenizer, cfg.max_length)
    val_ds   = PhraseBankDataset(df_val,   tokenizer, cfg.max_length)
    test_ds  = PhraseBankDataset(df_test,  tokenizer, cfg.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate,
                      weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )

    # Track best checkpoint by val macro-F1
    start = time.time()
    train_losses: List[float] = []
    best = {"val_f1": -1.0, "epoch": -1, "state_dict": None, "val_acc": 0.0}

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += out.loss.item()
        avg_loss = running / len(train_loader)
        train_losses.append(avg_loss)

        val_acc, val_f1, _, _ = _evaluate_loader(model, val_loader, device)
        if verbose:
            print(f"  Epoch {epoch}/{cfg.num_epochs}: "
                  f"train_loss={avg_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best["val_f1"]:
            best = {
                "val_f1": val_f1, "val_acc": val_acc, "epoch": epoch,
                "state_dict": {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()},
            }

    # Reload best-epoch weights and score the test set
    model.load_state_dict(best["state_dict"])
    test_acc, test_f1, _, y_pred = _evaluate_loader(model, test_loader, device)
    elapsed = time.time() - start

    if verbose:
        print(f"\nBest epoch: {best['epoch']}  Val macro-F1: {best['val_f1']:.4f}")
        print(f"Test accuracy:  {test_acc:.4f}")
        print(f"Test macro-F1:  {test_f1:.4f}")
        print(f"Elapsed: {elapsed:.1f}s")

    # Save predictions in the standard schema (sentence_id, predicted_label)
    preds_df = pd.DataFrame({
        "sentence_id": df_test["sentence_id"].values,
        "predicted_label": [ID2LABEL[int(p)] for p in y_pred],
    })
    Path(cfg.predictions_path).parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(cfg.predictions_path, index=False)

    metrics = {
        "config": asdict(cfg),
        "train_losses": train_losses,
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "best_val_f1":  best["val_f1"],
        "test_acc": test_acc,
        "test_f1":  test_f1,
        "elapsed_seconds": elapsed,
    }
    Path(cfg.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"Saved predictions to {cfg.predictions_path}")
        print(f"Saved metrics to {cfg.metrics_path}")

    if cfg.model_save_dir is not None:
        Path(cfg.model_save_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(cfg.model_save_dir)
        tokenizer.save_pretrained(cfg.model_save_dir)
    return metrics


def run_sweep(model_name: str = "ProsusAI/finbert",
              output_name: str = "finbert_finetuned",
              learning_rates: List[float] = (1e-5, 2e-5, 5e-5),
              num_epochs_list: List[int] = (3, 4),
              batch_size: int = 16,
              seed: int = 42,
              sweep_results_path: str = "results/tables/finbert_sweep.csv",
              ) -> pd.DataFrame:
    """Grid-search (lr x epochs). Picks the best config by val macro-F1 and
    retrains it once to produce the canonical predictions file."""
    # Make sure output dirs exist (otherwise plot/CSV saves later can fail)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    df_train, df_val, df_test = load_split()

    rows: List[Dict] = []
    best_val_f1 = -1.0
    best_cfg: Optional[TrainConfig] = None

    for lr in learning_rates:
        for n_ep in num_epochs_list:
            print(f"\n=== Sweep: lr={lr}, epochs={n_ep} ===")
            cfg = TrainConfig(
                model_name=model_name,
                output_name=output_name + f"_lr{lr}_ep{n_ep}",
                learning_rate=lr, num_epochs=n_ep,
                batch_size=batch_size, seed=seed,
                # Each sweep run goes to its own file so we can inspect later;
                # the canonical predictions path is overwritten below.
                predictions_path=f"predictions/sweep/{output_name}_lr{lr}_ep{n_ep}.csv",
                metrics_path=f"results/tables/sweep/{output_name}_lr{lr}_ep{n_ep}.json",
            )
            m = train_one_config(cfg, df_train, df_val, df_test, verbose=True)
            rows.append({
                "learning_rate": lr, "num_epochs": n_ep,
                "best_epoch_within_run": m["best_epoch"],
                "val_acc": m["best_val_acc"], "val_f1": m["best_val_f1"],
                "test_acc": m["test_acc"], "test_f1": m["test_f1"],
                "elapsed_s": round(m["elapsed_seconds"], 1),
            })
            if m["best_val_f1"] > best_val_f1:
                best_val_f1 = m["best_val_f1"]
                best_cfg = cfg

    sweep_df = pd.DataFrame(rows).sort_values("val_f1", ascending=False)
    Path(sweep_results_path).parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(sweep_results_path, index=False)
    print(f"\nSweep results saved to {sweep_results_path}")
    print(sweep_df.to_string(index=False))

    # Retrain the winning config and write it to the canonical output paths.
    print(f"\n=== Retraining best config for final predictions ===")
    print(f"Best config: lr={best_cfg.learning_rate}, epochs={best_cfg.num_epochs}")
    final_cfg = TrainConfig(
        model_name=best_cfg.model_name, output_name=output_name,
        learning_rate=best_cfg.learning_rate, num_epochs=best_cfg.num_epochs,
        batch_size=best_cfg.batch_size, seed=best_cfg.seed,
        predictions_path=f"predictions/{output_name}_predictions.csv",
        metrics_path=f"results/tables/{output_name}_metrics.json",
    )
    train_one_config(final_cfg, df_train, df_val, df_test, verbose=True)
    return sweep_df


def main():
    """Run a single FinBERT fine-tune with default hyperparameters."""
    train_one_config(TrainConfig())


if __name__ == "__main__":
    main()
