import os
import argparse
import numpy as np
from soma import aims

def save_nii(root):
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
                print(f"{sub_file} stats -> min: {vol_npy.min()}, max: {vol_npy.max()}, mean: {vol_npy.mean()}")
                # Convert to AIMS Volume
                vol_aims = aims.Volume(vol_npy)
                vol_aims.header()['voxel_size'] = [2.0, 2.0, 2.0]

                # Output file name
                output_nii = full_path.replace('.npy', '.nii.gz')
                aims.write(vol_aims, output_nii)
                print(f"    Saved to {output_nii}")

    except Exception as e:
        print(f"Exception occurred in folder: {root}")
        print(f"   Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Save nii files from npy files.")
    parser.add_argument('-p', '--path', type=str, help="Base folder path to the reconstructions.")
    args = parser.parse_args()

    if not args.path:
        print("No path provided. Please specify with -p.")
        return

    save_nii(args.path)


if __name__ == "__main__":
    main()