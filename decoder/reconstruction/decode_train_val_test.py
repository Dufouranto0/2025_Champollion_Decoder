import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from decoder.dataloader.dataloader import DataModule_Learning
from decoder.model.convnet import Decoder


def compute_loss(pred, target, loss_name="bce"):
    if loss_name == "bce":
        return F.binary_cross_entropy_with_logits(pred, target).item()
    elif loss_name == "mse":
        return F.mse_loss(pred, target).item()
    elif loss_name == "ce":
        return F.cross_entropy(pred, target).item()
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")


def compute_losses(model, dataloader, loss_name, device):
    results = []
    model.eval()
    with torch.no_grad():
        for latents, targets, subj_ids in dataloader:
            latents, targets = latents.to(device), targets.to(device)
            outputs = model(latents)
            for b in range(latents.size(0)):
                subj_id = subj_ids[b]
                pred = outputs[b]
                target = targets[b]
                loss = compute_loss(pred, target, loss_name)
                results.append((subj_id, loss, pred.cpu(), target.cpu()))
    return results


def compute_error_map(pred, target):
    return (pred - target) ** 2


def smooth_error_map(error_map, kernel_size=7):
    device = error_map.device
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=device)
    kernel = kernel / kernel.numel()
    smoothed = F.conv3d(error_map.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size // 2)
    return smoothed.squeeze(0).squeeze(0)


def build_population_stats(error_maps, max_subjects=500):
    if len(error_maps) > max_subjects:
        error_maps = error_maps[:max_subjects]
    stack = torch.stack(error_maps, dim=0)  # [N,D,H,W]
    mu = stack.mean(dim=0)
    sigma = stack.std(dim=0) + 1e-8
    return mu, sigma


def compute_population_zmaps(error_maps, mu, sigma):
    zmaps = []
    for err in error_maps:
        zmap = (err - mu) / sigma
        zmaps.append(zmap)
    return zmaps


def plot_loss_distributions(loss_dict, save_path):
    plt.figure(figsize=(8, 6))
    for split, losses in loss_dict.items():
        vals = [l for _, l, _, _ in losses]
        plt.hist(vals, bins=80, alpha=0.5, label=split, density=True)
    plt.xlabel("Loss")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Reconstruction Loss Distributions")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(run_dir, split="all", save_top_n=10, mode="worst", device="cuda", do_outliers=False):
    decoder_cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "decoder_config.yaml"))
    encoder_cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "encoder_config.yaml"))
    region = list(encoder_cfg.dataset.keys())[0]
    dataset_info = OmegaConf.to_container(encoder_cfg.dataset[region], resolve=True)

    dm = DataModule_Learning(decoder_cfg, dataset_info)
    dm.setup()

    splits = ["train", "val", "test"] if split == "all" else [split]

    # --- Init model ---
    example_loader = dm.test_dataloader()
    latent_dim = example_loader.dataset[0][0].shape[0]
    output_shape = example_loader.dataset[0][1].shape
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

    best_model_path = os.path.join(run_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_losses = {}
    recon_dir = os.path.join(run_dir, f"{mode}_reconstructions")
    os.makedirs(recon_dir, exist_ok=True)

    for sp in splits:
        loader = getattr(dm, f"{sp}_dataloader")()
        results = compute_losses(model, loader, decoder_cfg.loss, device)
        results.sort(key=lambda x: x[1], reverse=(mode == "worst"))
        all_losses[sp] = results

        # Save CSV
        df = pd.DataFrame([(sid, loss) for sid, loss, _, _ in results], columns=["Subject", "Loss"])
        df.to_csv(os.path.join(run_dir, f"{sp}_losses.csv"), index=False)
        print(f"Saved per-subject {sp} losses to {sp}_losses.csv")

        if do_outliers:
            smoothed_errors = []
            subj_ids = []
            for subj_id, _, pred, target in results:
                pred_sig = torch.sigmoid(pred) if decoder_cfg.loss == "bce" else pred
                pred_sig = pred_sig.squeeze(0).squeeze(0)
                err_map = compute_error_map(pred_sig, target.squeeze(0))
                smoothed = smooth_error_map(err_map)
                smoothed_errors.append(smoothed.cpu())
                subj_ids.append(subj_id)

            mu, sigma = build_population_stats(smoothed_errors, max_subjects=500)
            zmaps = compute_population_zmaps(smoothed_errors, mu, sigma)

            # save group stats
            np.save(os.path.join(run_dir, f"{sp}_error_mean.npy"), mu.numpy())
            np.save(os.path.join(run_dir, f"{sp}_error_std.npy"), sigma.numpy())
            print(f"Saved group mean/std error maps for {sp}")

            # save individual z-maps
            z_dir = os.path.join(run_dir, f"{sp}_zmaps")
            os.makedirs(z_dir, exist_ok=True)
            for subj_id, zmap in zip(subj_ids, zmaps):
                np.save(os.path.join(z_dir, f"{subj_id}_zmap.npy"), zmap.numpy())
            print(f"Saved {len(zmaps)} z-score maps for {sp}")

        # --- Save top-N reconstructions ---
        for subj_id, loss, pred, target in results[:save_top_n]:
            if decoder_cfg.loss == "mse":
                pred_np = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            elif decoder_cfg.loss == "bce":
                pred = torch.sigmoid(pred)
                pred_np = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            elif decoder_cfg.loss == "ce":
                pred_np = pred.squeeze(0)[1].cpu().numpy().astype(np.float32)
            else:
                raise ValueError(f"Unsupported loss: {decoder_cfg.loss}")

            pred_np = pred_np.transpose(2, 1, 0)
            np.save(os.path.join(recon_dir, f"{subj_id}_decoded.npy"), pred_np)

    if len(all_losses) > 1:
        plot_loss_distributions(all_losses, os.path.join(run_dir, "loss_distributions.png"))
        print("Saved loss distribution plot.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Path to run folder")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of reconstructions to save")
    parser.add_argument("-m", "--mode", choices=["best", "worst"], default="worst")
    parser.add_argument("--outliers", action="store_true", help="Compute voxelwise population z-scores")
    args = parser.parse_args()

    main(run_dir=args.path, split=args.split, save_top_n=args.num, mode=args.mode, do_outliers=args.outliers)
