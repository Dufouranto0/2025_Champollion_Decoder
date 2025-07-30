# train_utils.py

import torch
import torch.nn as nn
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-3, log_dir="runs/", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for latents, targets in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]"):
            latents, targets = latents.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(latents)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() * latents.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        val_loss = evaluate_model(model, val_loader, criterion, device)
        writer.add_scalar("Loss/Val", val_loss, epoch)

    writer.close()

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for latents, targets in val_loader:
            latents, targets = latents.to(device), targets.to(device)
            outputs = model(latents)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * latents.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss