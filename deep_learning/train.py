# train.py

import os
from omegaconf import OmegaConf
from dataloader.dataloader import DataModule_Learning
#from model.toy_model import ToyDecoderModel
from model.convnet import DecoderNet
from utils.train_utils import train_model
from reconstruction.save_npy import save_npy

def main():

    # Load configs
    decoder_cfg = OmegaConf.load("configs/config.yaml")
    
    assert os.path.exists(os.path.join(decoder_cfg.model_to_decode_path, \
                                              ".hydra", "config.yaml")), \
                        "Missing Hydra config file in model_to_decode_path"

    encoder_cfg = OmegaConf.load(os.path.join(decoder_cfg.model_to_decode_path, ".hydra", "config.yaml"))
    encoder_cfg["dataset_folder"] = decoder_cfg["dataset_folder"]
    region = list(encoder_cfg.dataset.keys())[0]
    dataset_info = OmegaConf.to_container(encoder_cfg.dataset[region], resolve=True)
    filters = encoder_cfg["filters"][::-1]

    # Setup data
    dm = DataModule_Learning(decoder_cfg, dataset_info)
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Init model
    print()
    print(region, '\n')

    latent_dim = train_loader.dataset[0][0].shape[0]
    print("latent_dim:", latent_dim, '\n')

    full_target = train_loader.dataset[0][1]
    if decoder_cfg["loss"] == "ce":
        # Target is (D, H, W) → output shape should be (C=2, D, H, W)
        output_shape = (2,) + full_target.shape
    else:
        # Target is already (C, D, H, W)
        output_shape = full_target.shape
    print("output_shape:", output_shape, '\n')

    model = DecoderNet(latent_dim=latent_dim, output_shape=output_shape, \
                       filters=filters, loss_name=decoder_cfg["loss"])
    #model = ToyDecoderModel(latent_dim=latent_dim, output_shape=output_shape)

    # Run training with TensorBoard logging
    log_dir = os.path.join("runs", decoder_cfg["experiment_name"])
    os.makedirs(log_dir, exist_ok=True)

    train_model(model, train_loader, val_loader,
                num_epochs=decoder_cfg["num_epochs"],
                lr=decoder_cfg["learning_rate"],
                loss_name=decoder_cfg["loss"],
                log_dir=log_dir,
                save_last=False,
                last_model_name="last_model.pt")


    recon_dir = os.path.join(log_dir, "reconstructions_epoch2")
    save_npy(model, val_loader, device="cuda", out_path=recon_dir, save_inputs=True, loss_name=decoder_cfg["loss"])

if __name__ == "__main__":
    main()