# decode_test_subjects.py

"""
Example:

python3 decode_test_subjects.py -p runs/Champollion_V1_after_ablation_256/58_fronto-parietal_medial_face_right_bce_0.0005 \
                                -n 5 \
                                -m best


Saved per-subject losses to runs/Champollion_V1_after_ablation_256/58_fronto-parietal_medial_face_right_bce_0.0005/test_losses.csv
[best] Saved sub-3097201 with loss 0.0378
[best] Saved sub-5234335 with loss 0.0394
[best] Saved sub-1607205 with loss 0.0394
[best] Saved sub-2510379 with loss 0.0402
[best] Saved sub-4124579 with loss 0.0415
Done.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from dataloader.dataloader import DataModule_Learning
from model.convnet import Decoder


def compute_loss(pred, target, loss_name="bce"):
    """Compute subject-level reconstruction loss."""
    if loss_name == "bce":
        return F.binary_cross_entropy_with_logits(pred, target).item()
    elif loss_name == "mse":
        return F.mse_loss(pred, target).item()
    elif loss_name == "ce":
        return F.cross_entropy(pred, target).item()
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")


def main(run_dir, save_top_n=10, mode="worst", device="cuda"):
    # --- Load configs ---
    decoder_cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "decoder_config.yaml"))
    encoder_cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "encoder_config.yaml"))

    region = list(encoder_cfg.dataset.keys())[0]
    dataset_info = OmegaConf.to_container(encoder_cfg.dataset[region], resolve=True)

    # --- Setup datamodule ---
    dm = DataModule_Learning(decoder_cfg, dataset_info)
    dm.setup()
    test_loader = dm.test_dataloader()

    # --- Init model ---
    latent_dim = test_loader.dataset[0][0].shape[0]
    output_shape = test_loader.dataset[0][1].shape
    filters = encoder_cfg["filters"][::-1]

    if decoder_cfg.loss == "ce":
        output_shape = (2,) + output_shape

    model = Decoder(
        latent_dim=latent_dim,
        output_shape=output_shape,
        filters=filters,
        drop_rate=decoder_cfg.dropout,
        loss_name=decoder_cfg.loss,
    ).to(device)

    # --- Load best weights ---
    best_model_path = os.path.join(run_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # --- Inference and loss calculation ---
    subject_losses = []
    recon_dir = os.path.join(run_dir, f"{mode}_reconstructions")
    os.makedirs(recon_dir, exist_ok=True)

    with torch.no_grad():
        for latents, targets, subj_ids in test_loader:
            latents, targets = latents.to(device), targets.to(device)
            outputs = model(latents)

            for b in range(latents.size(0)):
                subj_id = subj_ids[b]
                pred = outputs[b].unsqueeze(0)
                target = targets[b].unsqueeze(0)

                loss = compute_loss(pred, target, decoder_cfg.loss)
                subject_losses.append((subj_id, loss, pred.cpu(), target.cpu()))

    # --- Sort by loss ---
    subject_losses.sort(key=lambda x: x[1], reverse=(mode == "worst"))
    df = pd.DataFrame([(sid, loss) for sid, loss, _, _ in subject_losses],
                      columns=["Subject", "Loss"])
    df.to_csv(os.path.join(run_dir, "test_losses.csv"), index=False)
    print(f"Saved per-subject losses to {run_dir}/test_losses.csv")

    # --- Save reconstructions ---
    for subj_id, loss, pred, target in subject_losses[:save_top_n]:
        if decoder_cfg.loss == "bce":
            pred = torch.sigmoid(pred)
            pred_np = pred[0,0].numpy()
        elif decoder_cfg.loss == "mse":
            pred_np = pred[0,0].numpy()
        elif decoder_cfg.loss == "ce":
            pred_np = pred[0,1].numpy()
        else:
            raise ValueError(f"Unsupported loss: {decoder_cfg.loss}")

        # (D,H,W) → (Z,Y,X)
        pred_np = pred_np.transpose(2, 1, 0)

        # --- Save prediction ---
        np.save(os.path.join(recon_dir, f"{subj_id}_decoded.npy"), pred_np)

        # --- Save ground truth ---
        target_np = target[0].numpy()
        if target_np.ndim == 4:  # (C, D, H, W)
            target_np = target_np[0]
        target_np = target_np.transpose(2, 1, 0)
        np.save(os.path.join(recon_dir, f"{subj_id}_input.npy"), target_np)

        print(f"[{mode}] Saved {subj_id} with loss {loss:.4f}")

    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Path to run folder")
    parser.add_argument("-n", "--num", type=int, default=10,
                        help="Number of reconstructions to save")
    parser.add_argument("-m", "--mode", choices=["best", "worst"], default="worst",
                        help="Choose whether to save best or worst subjects")
    args = parser.parse_args()

    main(run_dir=args.path, save_top_n=args.num, mode=args.mode)
