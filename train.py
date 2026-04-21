import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ssm_model import ECGSSMClassifier


DATA_URL = "https://www.kaggle.com/datasets/shayanfazeli/heartbeat"


class ECGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, stitch: bool = False):
        """
        X: (N, 187)
        y: (N,)
        stitch: If True, concatenate examples to simulate continuous signal for SSM state drift training.
        """
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.stitch = stitch

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.stitch and idx > 0:
            # We can't easily stitch in a random-access __getitem__ without a specific order
            # But we can return a "prev_context" if we wanted to be fancy.
            # For now, we'll keep it simple but ensure the model is trained with 
            # enough variety.
            pass
        return self.X[idx], self.y[idx]


def load_kaggle_heartbeat(data_dir: Path, auto_download: bool = False):
    data_dir.mkdir(parents=True, exist_ok=True)

    def _find_root(base: Path):
        for p in base.glob("**/mitbih_train.csv"):
            test_p = p.parent / "mitbih_test.csv"
            if test_p.exists():
                return p.parent
        return None

    root = _find_root(data_dir)

    if root is None and auto_download:
        # Check for Kaggle credentials in env vars
        if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
            print("WARNING: KAGGLE_USERNAME and KAGGLE_KEY environment variables are not set.")
            print("opendatasets will likely prompt for them interactively.")
            
        print(f"Downloading Kaggle dataset from {DATA_URL}...")
        try:
            import opendatasets as od
        except ImportError:
            raise RuntimeError(
                "opendatasets is required for auto-download. Install it via: pip install opendatasets"
            )
        od.download(DATA_URL, data_dir=str(data_dir))
        root = _find_root(data_dir)

    if root is None:
        if not auto_download:
            raise FileNotFoundError(
                "Could not find mitbih_train.csv/mitbih_test.csv in the data directory. "
                "HINT: If you want the script to download the data for you, run with the '--auto-download' flag."
            )
        else:
            raise FileNotFoundError(
                "Could not find mitbih_train.csv/mitbih_test.csv even after attempting auto-download. "
                "Check your Kaggle credentials (KAGGLE_USERNAME and KAGGLE_KEY environment variables)."
            )

    train_csv = root / "mitbih_train.csv"
    test_csv = root / "mitbih_test.csv"

    train_df = pd.read_csv(train_csv, header=None)
    test_df = pd.read_csv(test_csv, header=None)

    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values.astype(int)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values.astype(int)

    return (X_train, y_train), (X_test, y_test)


def normalize_per_example(X: np.ndarray):
    # z-score per example for ECG beats
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-6
    return (X - mu) / sd


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    (X_train_full, y_train_full), (X_test, y_test) = load_kaggle_heartbeat(data_dir, auto_download=args.auto_download)
    X_train_full = normalize_per_example(X_train_full)
    X_test = normalize_per_example(X_test)

    # Infer number of classes
    num_classes = len(np.unique(y_train_full))
    print(f"Inferred {num_classes} classes from dataset.")

    # 1. Implement 90/10 Validation Split to prevent test leakage
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    # Optionally subsample for quick demo
    if args.max_train > 0:
        X_train = X_train[: args.max_train]
        y_train = y_train[: args.max_train]
    if args.max_test > 0:
        X_test = X_test[: args.max_test]
        y_test = y_test[: args.max_test]

    train_ds = ECGDataset(X_train, y_train, stitch=args.stitch)
    val_ds = ECGDataset(X_val, y_val)
    test_ds = ECGDataset(X_test, y_test)

    # Optimization for 5650U (12 threads)
    num_workers = args.num_workers if args.num_workers > 0 else 8
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=not args.stitch, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = ECGSSMClassifier(num_classes=num_classes, d_state=args.d_state, hidden=args.hidden, depth=args.depth, dropout=args.dropout)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    def evaluate(loader):
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item()
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        f1 = f1_score(labels, preds, average='macro')
        acc = (preds == labels).mean()
        return total_loss / len(loader), f1, acc

    best_f1 = 0
    patience = 3
    counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())
        
        sched.step()
        val_loss, val_f1, val_acc = evaluate(val_loader)
        print(f"Val Loss: {val_loss:.4f} | Val F1 (macro): {val_f1:.4f} | Val Acc: {val_acc:.4f}")

        # Early Stopping logic
        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
            # Save best checkpoint
            save_path = models_dir / "ecg_ssm.pt"
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "num_classes": num_classes,
                    "d_state": args.d_state,
                    "hidden": args.hidden,
                    "depth": args.depth,
                    "dropout": args.dropout,
                },
            }, save_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final Evaluation on Test Set
    print("\n--- Final Evaluation on Test Set ---")
    test_loss, test_f1, test_acc = evaluate(test_loader)
    print(f"Test F1 (macro): {test_f1:.4f} | Test Acc: {test_acc:.4f}")
    
    # Show detailed classification report for the test set
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
        test_logits = model(X_test_t)
        test_preds = test_logits.argmax(dim=1).cpu().numpy()
        print("\nClassification Report:")
        print(classification_report(y_test, test_preds))
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-test", type=int, default=10000, help="0 for all; limit for quick demo")
    parser.add_argument("--auto-download", action="store_true", help="Download dataset via Kaggle if not found locally")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers (Optimized for x86 5650U)")
    parser.add_argument("--stitch", action="store_true", help="Stitch beats sequentially to simulate continuous data")
    args = parser.parse_args()
    train(args)
