import os
import numpy as np
from soma import aims

# Set your root runs/ directory
runs_root = "runs/6_test_bce_5e-4"

# Recursively search for reconstructions_epoch* folders
for root, dirs, files in os.walk(runs_root):
    if os.path.basename(root).startswith("reconstructions_epoch"):
        print(f"Processing folder: {root}")

        try:
            for vol in ['_input', '_output']:
                npy_files = [f for f in os.listdir(root) if f.endswith(vol + '.npy')]
                print(f"Found {len(npy_files)} files for {vol}: {npy_files}")

                for sub_file in npy_files:
                    full_path = os.path.join(root, sub_file)
                    print(f"  -> Converting {sub_file}")

                    # Load and convert to float32
                    vol_npy = np.load(full_path).astype(np.float32)

                    # Convert to AIMS Volume
                    vol_aims = aims.Volume(vol_npy)
                    vol_aims.header()['voxel_size'] = [2.0, 2.0, 2.0]

                    # Output file name
                    output_nii = full_path.replace('.npy', '.nii.gz')
                    aims.write(vol_aims, output_nii)
                    print(f"    Saved to {output_nii}")

                    # Optional: remove original .npy
                    os.remove(full_path)
                    print(f"    Removed {full_path}")

        except Exception as e:
            print(f"Exception occurred in folder: {root}")
            print(f"   Error: {e}")