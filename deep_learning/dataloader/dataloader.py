# dataloader.py

import os
from typing import List, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from omegaconf import OmegaConf

"""
torch.random.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load decoder config
decoder_cfg = OmegaConf.load("../configs/config.yaml")

# Load encoder's training config
encoder_config_path = os.path.join(
    decoder_cfg.model_to_decode_path,
    ".hydra",
    "config.yaml"
)
encoder_cfg = OmegaConf.load(encoder_config_path)

# Override dataset_folder which conains a mistake
encoder_cfg["dataset_folder"] = decoder_cfg["dataset_folder"]

# Only resolve the dataset part to avoid errors in unrelated keys
region = list(encoder_cfg.dataset.keys())[0]
print("Region:", region)

dataset_info = OmegaConf.to_container(encoder_cfg.dataset[region], resolve=True)
"""

class LatentTargetDataset(Dataset):
    """
    PyTorch Dataset for loading latent vectors and corresponding target volumes.

    Args:
        latent_csv_path (str): Path to the latent vectors CSV file.
        target_npy_path (str): Path to the .npy file containing volume data.
        subjects_all_path (str): Path to CSV listing all subjects in the same order as .npy file.
        subject_list (List[str]): List of subject IDs to include in this dataset.
    """
    def __init__(self, latent_csv_path: str, target_npy_path: str,
                 subjects_all_path: str, subject_list: List[str]):

        self.subject_list = subject_list

        # Load latent vectors
        latent_df = pd.read_csv(latent_csv_path)
        latent_df = latent_df[latent_df["ID"].isin(subject_list)]
        latent_df = latent_df.set_index("ID").loc[subject_list].reset_index()
        self.latents = latent_df.drop(columns=["ID"])  # keep only the vector columns

        # Load subject order from the .csv (matching the .npy)
        subjects_all_df = pd.read_csv(subjects_all_path)
        subject_to_index = {subj: idx for idx, subj in enumerate(subjects_all_df["Subject"])}

        try:
            self.indices = [subject_to_index[subj] for subj in subject_list]
        except KeyError as e:
            raise ValueError(f"Subject {e.args[0]} not found in subjects_all list")

        # Load target volumes
        self.targets = np.load(target_npy_path)

        # Safety check
        assert len(self.latents) == len(self.indices), \
            "Mismatch between latent vectors and volume indices"

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent_vector = torch.tensor(self.latents.iloc[idx].values, dtype=torch.float32)
        target_volume = torch.tensor(self.targets[self.indices[idx]], dtype=torch.float32).permute(3, 0, 1, 2)
        return latent_vector, target_volume

class DataModule_Learning(LightningDataModule):
    """
    PyTorch Lightning DataModule for loading train, val, and test sets.

    Args:
        config (OmegaConf): Decoder configuration.
        dataset_info (dict): Dictionary with paths to dataset components.
    """
    def __init__(self, config, dataset_info):
        super().__init__()
        self.config = config
        self.dataset_info = dataset_info

    def setup(self, stage=None):

        # Full paths to CSV and NPY files
        train_path = os.path.join(self.config["model_to_decode_path"], self.config["train_csv"])
        val_test_path = os.path.join(self.config["model_to_decode_path"], self.config["val_test_csv"])
        subjects_all = os.path.join(self.config["dataset_folder"], self.dataset_info['subjects_all'])
        target_npy_path = os.path.join(self.config["dataset_folder"], self.dataset_info['numpy_all'])

        # Read subject splits
        train_data = pd.read_csv(train_path)
        val_test_data = pd.read_csv(val_test_path)
        val_test_data['IID'] = val_test_data['ID'].apply(lambda x: int(x[4:]))

        # Split into validation and test
        train_subjects = train_data['ID'].tolist()
        val_subjects = val_test_data[val_test_data['IID'] % 2 == 0]['ID'].tolist()
        test_subjects = val_test_data[val_test_data['IID'] % 2 == 1]['ID'].tolist()

        # Instantiate datasets
        self.dataset_train = LatentTargetDataset(
            latent_csv_path=train_path,
            target_npy_path=target_npy_path,
            subjects_all_path=subjects_all,
            subject_list=train_subjects
        )
        self.dataset_val = LatentTargetDataset(
            latent_csv_path=val_test_path,
            target_npy_path=target_npy_path,
            subjects_all_path=subjects_all,
            subject_list=val_subjects
        )
        self.dataset_test = LatentTargetDataset(
            latent_csv_path=val_test_path,
            target_npy_path=target_npy_path,
            subjects_all_path=subjects_all,
            subject_list=test_subjects
        )

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.config["batch_size"],
                          num_workers=self.config["num_cpu_workers"],
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.config["batch_size"],
                          num_workers=self.config["num_cpu_workers"],
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.config["batch_size"],
                          num_workers=self.config["num_cpu_workers"],
                          shuffle=False)