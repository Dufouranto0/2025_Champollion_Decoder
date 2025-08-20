# train_all_regions.py

import os, json
from omegaconf import OmegaConf
from dataloader.dataloader import DataModule_Learning
from model.convnet import DecoderNet
from utils.train_utils import train_model, evaluate_test
from train import load_configs, save_configs, infer_shapes, get_next_exp_number


def find_model_in_region(region_dir):
    """
    Find one model folder inside a region directory.
    Strategy: pick the first subdirectory that has a .hydra/config.yaml
    """
    for candidate in sorted(os.listdir(region_dir)):
        cand_path = os.path.join(region_dir, candidate)
        if os.path.isdir(cand_path) and os.path.exists(os.path.join(cand_path, ".hydra", "config.yaml")):
            return cand_path
    return None


def train_all_regions(config_path="configs/config.yaml", out_json="all_results.json"):
    # --- Load base decoder config ---
    decoder_cfg = OmegaConf.load(config_path)

    model_parent_dir = decoder_cfg.model_to_decode_dir
    if not os.path.isdir(model_parent_dir):
        raise NotADirectoryError(f"{model_parent_dir} is not a valid directory")

    results = {}

    # --- Loop over region subfolders ---
    for region in sorted(os.listdir(model_parent_dir)):
        region_dir = os.path.join(model_parent_dir, region)
        if not os.path.isdir(region_dir):
            continue

        model_path = find_model_in_region(region_dir)
        if model_path is None:
            print(f"Skipping {region}: no valid model found")
            continue

        print(f"\n=== Training decoder for region: {region} ===")
        print(f"Using encoder model: {model_path}")

        # Temporarily inject model_to_decode_path for this run
        decoder_cfg.model_to_decode_path = model_path

        # --- Load corresponding encoder config ---
        encoder_config_path = os.path.join(model_path, ".hydra", "config.yaml")
        encoder_cfg = OmegaConf.load(encoder_config_path)
        encoder_cfg["dataset_folder"] = decoder_cfg["dataset_folder"]

        dataset_info = OmegaConf.to_container(encoder_cfg.dataset[region], resolve=True)
        dm = DataModule_Learning(decoder_cfg, dataset_info)
        dm.setup()

        latent_dim, output_shape, filters, _ = infer_shapes(dm, decoder_cfg, encoder_cfg)

        model = DecoderNet(
            latent_dim=latent_dim,
            output_shape=output_shape,
            filters=filters,
            loss_name=decoder_cfg.loss,
            drop_rate=decoder_cfg.dropout,
        )

        # --- Training ---
        nb_exp = get_next_exp_number(decoder_cfg.log_dir)
        experiment_name = f"{nb_exp}_{region}_{decoder_cfg.loss}_{decoder_cfg.learning_rate}"
        log_dir = os.path.join(decoder_cfg.log_dir, experiment_name)
        os.makedirs(log_dir, exist_ok=True)

        save_configs(log_dir, decoder_cfg, encoder_cfg)

        model, history = train_model(
            model,
            dm.train_dataloader(),
            dm.val_dataloader(),
            num_epochs=decoder_cfg.num_epochs,
            lr=decoder_cfg.learning_rate,
            loss_name=decoder_cfg.loss,
            log_dir=log_dir,
            save_best_model=decoder_cfg.save_best_model,
        )

        # --- Evaluate on test set ---
        test_loss = evaluate_test(model, dm.test_dataloader(), loss_name=decoder_cfg.loss)
        print(f"[{region}] Test loss: {test_loss:.4f}")

        results[region] = {
            "test_loss": test_loss,
            "history": history,
            "model_path": model_path,
        }

    # --- Save all results to JSON ---
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results for all regions in {out_json}")


if __name__ == "__main__":
    train_all_regions()
