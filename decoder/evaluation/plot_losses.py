# evaluation/plot_losses.py

import os, json, matplotlib.pyplot as plt

def plot_loss_curve(log_dir, out_path):
    with open(os.path.join(log_dir, "loss_history.json")) as f:
        history = json.load(f)

    plt.figure()
    plt.plot(history["train_loss"], label="Train BCE")
    plt.plot(history["val_loss"], label="Val BCE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(os.path.basename(log_dir))
    plt.legend()
    plt.savefig(out_path)
    plt.close()
