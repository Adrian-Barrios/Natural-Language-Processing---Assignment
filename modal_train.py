"""Train BiLSTM + Linear Attention on Modal GPU. Saves model.pth + vocab.json locally."""
import json
from pathlib import Path

import modal

app = modal.App("nlp-bilstm-attn")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "pandas", "numpy", "scikit-learn")
    .add_local_dir("./data", "/data", copy=True)
)


@app.function(image=image, gpu="T4", timeout=3600)
def train(epochs: int = 15, max_len: int = 50, batch_size: int = 128):
    import random
    import re
    from collections import Counter

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset

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

    def tokenize(text):
        return re.findall(r"\b\w+\b", str(text).lower())

    counts = Counter(t for txt in train_df["text"] for t in tokenize(txt))
    vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(20000))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    def encode(tokens):
        ids = [vocab.get(w, 1) for w in tokens][:max_len]
        if not ids:
            ids = [1]  # <UNK> fallback so attention mask has ≥1 valid position
        return ids + [0] * (max_len - len(ids))

    def augment(tokens, p_drop=0.1, p_swap=0.1):
        if len(tokens) > 1:
            kept = [t for t in tokens if random.random() > p_drop]
            if kept:
                tokens = kept
        if len(tokens) > 1 and random.random() < p_swap:
            i = random.randint(0, len(tokens) - 2)
            tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
        return tokens

    class EmotionDataset(Dataset):
        def __init__(self, df, is_test=False, do_augment=False):
            self.tokens = [tokenize(t) for t in df["text"].tolist()]
            self.is_test = is_test
            self.do_augment = do_augment
            if not is_test:
                self.labels = df[emotion_cols].values.astype(np.float32)

        def __len__(self):
            return len(self.tokens)

        def __getitem__(self, idx):
            toks = list(self.tokens[idx])
            if self.do_augment:
                toks = augment(toks)
            x = torch.tensor(encode(toks), dtype=torch.long)
            if self.is_test:
                return x
            return x, torch.tensor(self.labels[idx])

    class LinearAttention(nn.Module):
        """Additive attention with a learnable linear scoring layer."""

        def __init__(self, hidden_dim):
            super().__init__()
            self.attn = nn.Linear(hidden_dim, 1)

        def forward(self, lstm_out, mask):
            # If a row has no valid tokens, fall back to attending to position 0
            # so softmax never sees an all -inf row (which would produce NaN).
            safe_mask = mask.clone()
            safe_mask[:, 0] |= ~mask.any(dim=1)
            scores = self.attn(lstm_out).squeeze(-1)
            scores = scores.masked_fill(~safe_mask, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            return (weights * lstm_out).sum(dim=1)

    class BiLSTMAttentionModel(nn.Module):
        def __init__(self, vocab_size, embed_dim=200, hidden_dim=256, output_dim=28,
                     dropout_p=0.4, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.dropout = nn.Dropout(dropout_p)
            self.lstm = nn.LSTM(
                embed_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_p if num_layers > 1 else 0.0,
            )
            self.attention = LinearAttention(hidden_dim * 2)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)

        def forward(self, x):
            mask = x != 0
            embedded = self.dropout(self.embedding(x))
            lstm_out, _ = self.lstm(embedded)
            context = self.attention(lstm_out, mask)
            return self.fc(self.dropout(context))

    train_split, val_split = train_test_split(train_df, test_size=0.1, random_state=SEED)
    train_loader = DataLoader(EmotionDataset(train_split, do_augment=True),
                              batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(EmotionDataset(val_split),
                            batch_size=batch_size, num_workers=2)

    model = BiLSTMAttentionModel(vocab_size, output_dim=num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                preds.append(torch.sigmoid(logits).cpu().numpy())
                targets.append(y.numpy())
        auc = roc_auc_score(np.vstack(targets), np.vstack(preds), average="macro")
        scheduler.step(auc)
        print(f"Epoch {epoch:02d} | loss {train_loss:.4f} | val ROC-AUC {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ↑ new best ({best_auc:.4f})")

    print(f"\nBest val ROC-AUC: {best_auc:.4f}")

    buf = Path("/tmp/model.pth")
    torch.save(best_state, buf)
    model_bytes = buf.read_bytes()
    return model_bytes, vocab, best_auc


@app.local_entrypoint()
def main(epochs: int = 15):
    model_bytes, vocab, best_auc = train.remote(epochs=epochs)
    out_dir = Path("deliverable_folder")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "model_best.pth").write_bytes(model_bytes)
    (out_dir / "vocab.json").write_text(json.dumps(vocab))
    print(f"Saved deliverable_folder/model_best.pth  (val ROC-AUC={best_auc:.4f})")
    print(f"Saved deliverable_folder/vocab.json")
