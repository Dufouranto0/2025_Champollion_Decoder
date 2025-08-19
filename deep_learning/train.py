# train.py

import os
from omegaconf import OmegaConf
from dataloader.dataloader import DataModule_Learning
from model.convnet import DecoderNet
from utils.train_utils import train_model
from reconstruction.save_npy import save_npy
import shutil
import re



def load_configs(decoder_cfg_path: str):
    """
    Load decoder config and corresponding encoder config.
    """
    decoder_cfg = OmegaConf.load(decoder_cfg_path)

    encoder_config_path = os.path.join(
        decoder_cfg.model_to_decode_path, ".hydra", "config.yaml"
    )
    if not os.path.exists(encoder_config_path):
        raise FileNotFoundError(
            f"Missing Hydra config file: {encoder_config_path}"
        )

    encoder_cfg = OmegaConf.load(encoder_config_path)
    encoder_cfg["dataset_folder"] = decoder_cfg["dataset_folder"]

    return decoder_cfg, encoder_cfg

def save_configs(log_dir, decoder_cfg, encoder_cfg):
    """Save decoder and encoder configs in Hydra-style directory."""
    hydra_dir = os.path.join(log_dir, ".hydra")
    os.makedirs(hydra_dir, exist_ok=True)

    # Save the main config
    OmegaConf.save(config=decoder_cfg, f=os.path.join(hydra_dir, "config.yaml"))

    # Optionally also save the encoder config for traceability
    OmegaConf.save(config=encoder_cfg, f=os.path.join(hydra_dir, "encoder_config.yaml"))

    # Save overrides if needed (e.g., command line args)
    #with open(os.path.join(hydra_dir, "overrides.yaml"), "w") as f:
    #    f.write("# Add CLI overrides here if used\n")

    # Save a copy of the config.yaml used to launch the job (for convenience)
    #shutil.copy("configs/config.yaml", os.path.join(hydra_dir, "original_config.yaml"))

def infer_shapes(dm: DataModule_Learning, decoder_cfg, encoder_cfg):
    """
    Infer latent dimension, output shape, and filters from dataloader + configs.
    """
    train_loader = dm.train_dataloader()

    region = list(encoder_cfg.dataset.keys())[0]
    filters = encoder_cfg["filters"][::-1]

    latent_dim = train_loader.dataset[0][0].shape[0]
    full_target = train_loader.dataset[0][1]

    if decoder_cfg["loss"] == "ce":
        # target is (D, H, W) → output should be (C=2, D, H, W)
        output_shape = (2,) + full_target.shape
    else:
        output_shape = full_target.shape

    return latent_dim, output_shape, filters, region


def get_next_exp_number(runs_dir="runs"):
    os.makedirs(runs_dir, exist_ok=True)  # make sure folder exists
    exp_folders = [f for f in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, f))]
    
    numbers = []
    for folder in exp_folders:
        match = re.match(r"^(\d+)", folder)  # match leading digits
        if match:
            numbers.append(int(match.group(1)))
    
    if numbers:
        return max(numbers) + 1
    else:
        return 1  # if no experiment exists yet


def main():
    # --- Load configs ---
    decoder_cfg, encoder_cfg = load_configs("configs/config.yaml")
    region = list(encoder_cfg.dataset.keys())[0]

    print("Region:", region)

    dataset_info = OmegaConf.to_container(
        encoder_cfg.dataset[region], resolve=True
    )

    # --- Setup data ---
    dm = DataModule_Learning(decoder_cfg, dataset_info)
    dm.setup()

    latent_dim, output_shape, filters, region = infer_shapes(
        dm, decoder_cfg, encoder_cfg
    )

    print("latent_dim:", latent_dim)
    print("output_shape:", output_shape)

    loss = decoder_cfg.loss
    print("loss_name:", loss) 


    # --- Init model ---
    model = DecoderNet(
        latent_dim=latent_dim,
        output_shape=output_shape,
        filters=filters,
        loss_name=loss,
        drop_rate=decoder_cfg.dropout,
    )

    # --- Training ---
    nb_exp = get_next_exp_number("runs")
    experiment_name = f"{nb_exp}_{region}_{loss}_{decoder_cfg.learning_rate}"
    log_dir = os.path.join("runs", experiment_name)
    
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")

    # --- Save configs ---
    save_configs(log_dir, decoder_cfg, encoder_cfg)

    train_model(
        model,
        dm.train_dataloader(),
        dm.val_dataloader(),
        num_epochs=decoder_cfg.num_epochs,
        lr=decoder_cfg.learning_rate,
        loss_name=loss,
        log_dir=log_dir,
        save_best_model=decoder_cfg.save_best_model,
    )

    # --- Reconstructions ---
    recon_dir = os.path.join(log_dir, f"reconstructions_epoch{decoder_cfg.num_epochs}")
    save_npy(
        model,
        dm.val_dataloader(),
        device="cuda",
        out_path=recon_dir,
        save_inputs=True,
        loss_name=loss,
    )


if __name__ == "__main__":
    main()
