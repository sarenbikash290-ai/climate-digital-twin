import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import json

from src.model.dataset import MaharashtraClimateDataset
from src.model.convlstm import ConvLSTM


def train_model(
    years      = list(range(2018, 2024)),
    seq_len    = 7,
    pred_len   = 3,
    batch_size = 4,
    epochs     = 30,
    lr         = 1e-3,
    save_dir   = "models/"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    start_epoch     = 1
    best_val_loss   = float("inf")
    history         = {"train_loss": [], "val_loss": []}
    print(f"🖥️  Training on: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = MaharashtraClimateDataset(years=years, seq_len=seq_len, pred_len=pred_len)

    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n📊 Dataset split:")
    print(f"   Train sequences: {train_size}")
    print(f"   Val   sequences: {val_size}")

    # ── Model ─────────────────────────────────────────────────────────────────
    # Get spatial dims from dataset
    sample_x, _ = dataset[0]
    _, _, H, W  = sample_x.shape

    model = ConvLSTM(
        in_channels     = 3,
        hidden_channels = 64,
        kernel_size     = 3,
        num_layers      = 2,
        pred_len        = pred_len,
        out_channels    = 3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    # ── Resume from checkpoint if exists ──────────────────────────────────
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 Resuming from checkpoint...")
        ckpt          = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        history       = ckpt["history"]
        print(f"   Resumed from epoch {ckpt['epoch']} | Best val loss: {best_val_loss:.6f}")
    else:
        print(f"\n🆕 Starting fresh training...")
    print(f"\n🧠 Model: ConvLSTM")
    print(f"   Spatial dims    : {H} × {W}")
    print(f"   Total params    : {total_params:,}")

    # ── Training setup ────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()


    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n🚀 Starting training for {epochs} epochs...\n")

    for epoch in range(start_epoch, epochs + 1):
        # Train
        model.train()
        train_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            tag = "✅ (best)"
        else:
            tag = ""

        print(f"Epoch [{epoch:03d}/{epochs}] "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} {tag}")
        # Save checkpoint every epoch
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss":   best_val_loss,
            "history":         history
        }, checkpoint_path)

    # ── Save training history ──────────────────────────────────────────────────
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\n🎉 Training complete!")
    print(f"   Best Val Loss : {best_val_loss:.6f}")
    print(f"   Model saved   : {save_dir}best_model.pth")
    print(f"   History saved : {save_dir}history.json")

    return model, history


if __name__ == "__main__":
    train_model()