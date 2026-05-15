"""Fine-tune DistilBERT for Reddit GoEmotions (28-label) on a Modal GPU.

Outputs:
  - deliverable_folder/model_best.pth   (fine-tuned state_dict)
  - deliverable_folder/tokenizer/       (saved tokenizer + base config)

Run:  python -m modal run modal_train_bert.py
"""
import io
import tarfile
from pathlib import Path

import modal

PRETRAINED = "distilbert-base-uncased"

app = modal.App("nlp-distilbert")


def _prefetch():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    AutoTokenizer.from_pretrained(PRETRAINED)
    AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED, num_labels=28, problem_type="multi_label_classification"
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "pandas",
        "numpy",
        "scikit-learn",
    )
    .add_local_dir("./data", "/data", copy=True)
    .run_function(_prefetch)
)


@app.function(image=image, gpu="T4", timeout=3600)
def train(epochs: int = 3, max_len: int = 128, batch_size: int = 32, lr: float = 2e-5):
    import random

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = pd.read_csv("/data/train_kaggle.csv")
    test_df = pd.read_csv("/data/test_kaggle.csv")
    emotion_cols = [c for c in train_df.columns if c not in ("ID", "text")]
    num_labels = len(emotion_cols)
    print(f"Train={len(train_df)}  Test={len(test_df)}  Labels={num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)

    class EmotionDataset(Dataset):
        def __init__(self, df, is_test=False):
            enc = tokenizer(
                df["text"].astype(str).tolist(),
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.is_test = is_test
            if not is_test:
                self.labels = torch.tensor(
                    df[emotion_cols].values.astype(np.float32), dtype=torch.float32
                )

        def __len__(self):
            return self.input_ids.size(0)

        def __getitem__(self, idx):
            item = {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
            }
            if not self.is_test:
                item["labels"] = self.labels[idx]
            return item

    train_split, val_split = train_test_split(
        train_df, test_size=0.1, random_state=SEED
    )
    train_loader = DataLoader(
        EmotionDataset(train_split), batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        EmotionDataset(val_split), batch_size=batch_size, num_workers=2
    )
    test_loader = DataLoader(
        EmotionDataset(test_df, is_test=True), batch_size=batch_size, num_workers=2
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    ).to(device)

    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler()

    best_auc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * input_ids.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.cuda.amp.autocast():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds.append(torch.sigmoid(logits.float()).cpu().numpy())
                targets.append(batch["labels"].numpy())
        auc = roc_auc_score(np.vstack(targets), np.vstack(preds), average="macro")
        print(f"Epoch {epoch:02d} | loss {train_loss:.4f} | val ROC-AUC {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ↑ new best ({best_auc:.4f})")

    print(f"\nBest val ROC-AUC: {best_auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.cuda.amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            test_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    submission = pd.DataFrame(np.vstack(test_probs), columns=emotion_cols)
    submission.insert(0, "ID", test_df["ID"].values)
    submission_csv = submission.to_csv(index=False).encode("utf-8")
    print(f"Built submission for {len(submission)} test rows")

    state_buf = io.BytesIO()
    torch.save(best_state, state_buf)

    tok_dir = Path("/tmp/tokenizer")
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tok_dir)
    # Save the base config too so inference can rebuild the architecture offline.
    model.config.save_pretrained(tok_dir)

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tar:
        tar.add(tok_dir, arcname=".")

    return state_buf.getvalue(), tar_buf.getvalue(), submission_csv, best_auc


@app.local_entrypoint()
def main(epochs: int = 3):
    model_bytes, tokenizer_tar, submission_csv, best_auc = train.remote(epochs=epochs)

    out_dir = Path("deliverable_folder")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "model_best.pth").write_bytes(model_bytes)
    (out_dir / "submission.csv").write_bytes(submission_csv)

    tok_out = out_dir / "tokenizer"
    tok_out.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tokenizer_tar), mode="r") as tar:
        tar.extractall(tok_out)

    print(f"Saved deliverable_folder/model_best.pth  (val ROC-AUC={best_auc:.4f})")
    print(f"Saved deliverable_folder/tokenizer/")
    print(f"Saved deliverable_folder/submission.csv")
