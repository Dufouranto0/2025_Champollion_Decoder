# save_npy.py

import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def save_npy(
    model,
    dataloader,
    device,
    out_path,
    save_inputs: bool = False,
    loss_name: str = "bce",
    max_batches_to_save: int = 2,
    save_histograms: bool = False,
):
    """
    Save model reconstructions as .npy files and optionally input volumes.
    Also saves histograms of predicted values (before thresholding).

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader yielding (latent, target, subj_id).
        device: torch device.
        out_path (str): Directory to save results.
        save_inputs (bool): If True, save input volumes as well.
        loss_name (str): "mse", "bce", or "ce".
        max_batches_to_save (int): How many batches of outputs to save.
        save_histograms (bool): Save histograms of continuous predictions.
    """
    model.eval()
    os.makedirs(out_path, exist_ok=True)
    hist_dir = os.path.join(out_path, "histograms")
    if save_histograms:
        os.makedirs(hist_dir, exist_ok=True)

    with torch.no_grad():
        for i, (latents, targets, subj_ids) in enumerate(dataloader):
            if i >= max_batches_to_save:
                break

            latents, targets = latents.to(device), targets.to(device)
            outputs = model(latents)

            for b in range(latents.size(0)):
                subj_id = subj_ids[b]  # real subject ID from dataloader

                # --------- Predicted volume ---------
                if loss_name == "mse":
                    # Continuous model output (no threshold)
                    raw = outputs[b]                     # shape: (1, D, H, W)
                    values = raw.cpu().numpy()           
                    decoded_vol = values[0].astype(np.float32)

                elif loss_name == "bce":
                    # Continuous probabilities in [0,1]
                    raw = torch.sigmoid(outputs[b])      # shape: (1, D, H, W)
                    values = raw.cpu().numpy()           
                    decoded_vol = values[0].astype(np.float32)

                elif loss_name == "ce":
                    probs = outputs[b]       # (C, D, H, W)
                    pred = probs[1,:,:,:].cpu().numpy() # (1, D, H, W)
                    decoded_vol = pred.astype(np.float32)

                else:
                    raise ValueError(f"Unsupported loss function: {loss_name}")

                # Reorder axes: (D, H, W) --> (Z, Y, X)
                decoded_vol = decoded_vol.transpose(2, 1, 0)

                # Save npy
                decoded_path = os.path.join(out_path, f"{subj_id}_decoded.npy")
                np.save(decoded_path, decoded_vol)

                # --------- Histogram of raw values ---------
                if save_histograms and loss_name in ["mse", "bce"]:
                    plt.figure(figsize=(6, 4))
                    plt.hist(values.ravel(), bins=50, color="steelblue", alpha=0.7)
                    plt.title(f"Predicted value distribution - {subj_id}")
                    plt.xlabel("Predicted value (before threshold)")
                    plt.ylabel("Voxel count")
                    plt.grid(True, alpha=0.3)
                    fig_path = os.path.join(hist_dir, f"{subj_id}_hist.png")
                    plt.savefig(fig_path)
                    plt.close()

                # --------- Save input/target volume ---------
                if save_inputs:
                    input_vol = targets[b].cpu().numpy().astype(np.float32)
                    if input_vol.ndim == 4:  # (C, D, H, W)
                        input_vol = input_vol[0]
                    input_vol = input_vol.transpose(2, 1, 0)
                    input_path = os.path.join(out_path, f"{subj_id}_input.npy")
                    np.save(input_path, input_vol)
