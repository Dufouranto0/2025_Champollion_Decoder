# utils/train_utils.py

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
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
    lr: float = 5e-4,
    loss_name: str = "bce",
    out_dir: str = "runs",
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
        out_dir (str): path for tensorboard logs
        save_best_model (bool): whether to save the best model checkpoint
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)

    loss_fn = get_loss_fn(loss_name)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=out_dir)
    best_val_loss = float("inf")

    history = {"train_loss": [], "val_loss": []}

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
        history["train_loss"].append(avg_train_loss)
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
        history["val_loss"].append(avg_val_loss)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # Save best model
        if save_best_model and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(out_dir, exist_ok=True)
            best_model_path = os.path.join(out_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch+1} with val_loss={avg_val_loss:.4f}")
    
    # --- Save history as JSON
    with open(os.path.join(out_dir, "loss_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    writer.close()
    
    return model, history

def evaluate_test(model, test_loader, loss_name="bce"):
    """
    Evaluate a trained model on the test set and return average loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loss_fn = get_loss_fn(loss_name)
    total_loss = 0.0
    with torch.no_grad():
        for latents, targets, _ in test_loader:
            latents, targets = latents.to(device), targets.to(device)
            outputs = model(latents)
            total_loss += loss_fn(outputs, targets).item()

    return total_loss / len(test_loader)

