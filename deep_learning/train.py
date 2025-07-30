# train.py

import os
from omegaconf import OmegaConf
from dataloader.dataloader import DataModule_Learning
#from model.toy_model import ToyDecoderModel
from model.convnet import DecoderNet
from utils.train_utils import train_model

def main():
    # Load configs
    decoder_cfg = OmegaConf.load("/volatile/ad279118/2025_Champollion_Decoder/deep_learning/configs/config.yaml")
    encoder_cfg = OmegaConf.load(os.path.join(decoder_cfg.model_to_decode_path, ".hydra", "config.yaml"))
    encoder_cfg["dataset_folder"] = decoder_cfg["dataset_folder"]
    region = list(encoder_cfg.dataset.keys())[0]
    dataset_info = OmegaConf.to_container(encoder_cfg.dataset[region], resolve=True)

    # Setup data
    dm = DataModule_Learning(decoder_cfg, dataset_info)
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Init model
    latent_dim = train_loader.dataset[0][0].shape[0]
    print("latent_dim:", latent_dim)

    output_shape = train_loader.dataset[0][1].shape  # (C, D, H, W)
    print("output_shape:", output_shape)

    model = DecoderNet(latent_dim=latent_dim, output_shape=output_shape)

    # Run training with TensorBoard logging
    log_dir = os.path.join("runs", decoder_cfg["experiment_name"])
    os.makedirs(log_dir, exist_ok=True)

    train_model(model, train_loader, val_loader,
                num_epochs=decoder_cfg.get("epochs", 80),
                lr=decoder_cfg.get("lr", 1e-3),
                log_dir=log_dir)

if __name__ == "__main__":
    main()