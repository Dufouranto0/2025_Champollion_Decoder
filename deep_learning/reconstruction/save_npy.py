import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def save_npy(model, dataloader, device, out_path, save_inputs=False, loss_name='ce'):
    model.eval()
    os.makedirs(out_path, exist_ok=True)
    limit = 2  # how many batches to save

    with torch.no_grad():
        for i, (latents, targets) in enumerate(dataloader):
            if i >= limit:
                break

            latents, targets = latents.to(device), targets.to(device)
            outputs = model(latents)

            for b in range(latents.size(0)):
                subj_id = f"subject_{i * dataloader.batch_size + b:04d}"

                # Get predicted volume as binary float32 (0.0 / 1.0)
                if loss_name == 'mse':
                    pred = outputs[b]
                    print(pred.shape)
                    pred = (pred > 0.5).float()
                    output_vol = pred.cpu().numpy().astype(np.float32)[0]  # remove channel dim
                elif loss_name == 'bce':
                    pred = torch.sigmoid(outputs[b])
                    pred = (pred > 0.5).float()
                    output_vol = pred.cpu().numpy().astype(np.float32)[0]
                elif loss_name == 'ce':
                    pred = torch.argmax(outputs[b], dim=0).float()
                    output_vol = pred.cpu().numpy().astype(np.float32)
                else:
                    raise ValueError(f"Unsupported loss function: {loss_name}")

                # Transpose from (D, H, W) to (X, Y, Z) as (Z, Y, X)
                output_vol = output_vol.transpose(2, 1, 0)

                # Save output
                output_path = os.path.join(out_path, f"{subj_id}_output.npy")
                np.save(output_path, output_vol)

                # Optionally save input/target
                if save_inputs:
                    input_vol = targets[b].cpu().numpy().astype(np.float32)
                    if input_vol.ndim == 4:
                        input_vol = input_vol[0]
                    input_vol = input_vol.transpose(2, 1, 0)
                    input_path = os.path.join(out_path, f"{subj_id}_input.npy")
                    np.save(input_path, input_vol)