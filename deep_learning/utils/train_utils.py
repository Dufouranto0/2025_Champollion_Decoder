# utils/train_utils.py

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


def get_loss_fn(loss_name: str):
    """Return appropriate loss function based on name."""

    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    loss_name: str = "bce",
    log_dir: str = "runs/decoder",
    save_best_model: bool = True,
):
    """
    Train the model with given loaders.

    Args:
        model: torch.nn.Module
        train_loader: DataLoader yielding (latents, targets, subj_ids)
        val_loader: DataLoader yielding (latents, targets, subj_ids)
        num_epochs (int)
        lr (float)
        loss_name (str): one of {"mse", "bce", "ce"}
        log_dir (str): path for tensorboard logs
        save_best_model (bool): whether to save the best model checkpoint
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)

    loss_fn = get_loss_fn(loss_name)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=log_dir)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # -----------------------
        # Training
        # -----------------------
        model.train()
        total_train_loss = 0.0

        for latents, targets, subj_ids in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]"):
            latents, targets = latents.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(latents)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # -----------------------
        # Validation
        # -----------------------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for latents, targets, subj_ids in val_loader:
                latents, targets = latents.to(device), targets.to(device)
                outputs = model(latents)
                loss = loss_fn(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # Save best model
        if save_best_model and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(log_dir, exist_ok=True)
            best_model_path = os.path.join(log_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch+1} with val_loss={avg_val_loss:.4f}")

    writer.close()
    return model
