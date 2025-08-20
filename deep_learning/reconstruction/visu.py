#visu.py
"""
bv bash

cd 2025_Champollion_Decoder/deep_learning

python3 reconstruction/visu.py \
  -p example \
  -l bce \
  -s sub-1110622_input.nii.gz,sub-1150302_input.nii.gz
"""


import anatomist.api as ana
from soma.qt_gui.qt_backend import Qt
from soma import aims
import numpy as np
import argparse
import os
import glob

def build_gradient(pal):
    """Builds a gradient palette."""
    gw = ana.cpp.GradientWidget(
        None, 'gradientwidget',
        pal.header()['palette_gradients'])
    gw.setHasAlpha(True)
    nc = pal.shape[0]
    rgbp = gw.fillGradient(nc, True)
    rgb = rgbp.data()
    npal = pal.np['v']
    pb = np.frombuffer(rgb, dtype=np.uint8).reshape((nc, 4))
    npal[:, 0, 0, 0, :] = pb
    npal[:, 0, 0, 0, :3] = npal[:, 0, 0, 0,
                                :3][:, ::-1]  # Convert BGRA to RGBA
    pal.update()

def plot_ana(recon_dir, n_subjects_to_display, loss_name, listsub):

    referential1 = a.createReferential()

    if listsub:
        input_files = [os.path.join(recon_dir, sub) for sub in listsub]

    else:
        print("No list of subjects given in argument, take 4 random subjects.")
        # Find sub-XXXX_input.nii.gz 
        input_files = glob.glob(os.path.join(recon_dir, "sub-*_input.nii.gz"))

        # Limit to N subjects
        numbers = np.random.choice(len(input_files), size=n_subjects_to_display, replace=False)
        input_files = [input_files[i] for i in numbers]
        print("Input files:")
        print([os.path.basename(file) for file in input_files])

    for i, input_path in enumerate(input_files):
        subject_id = os.path.basename(input_path).split('_input')[0]
        output_path = os.path.join(recon_dir, f"{subject_id}_output.nii.gz")

        if not os.path.isfile(output_path):
            print(f"Missing output for {subject_id}, skipping.")
            continue

        # Read input and output volumes
        input_vol = aims.read(input_path)
        output_vol = aims.read(output_path)

        # Display input
        dic_windows[f'a_input_{i}'] = a.toAObject(input_vol)
        dic_windows[f'r_input_{i}'] = a.fusionObjects(objects=[dic_windows[f'a_input_{i}']],
                                                    method='VolumeRenderingFusionMethod')
        dic_windows[f'r_input_{i}'].releaseAppRef()
        dic_windows[f'r_input_{i}'].assignReferential(referential1)
        dic_windows[f'w_input_{i}'] = a.createWindow('3D', block=block)
        dic_windows[f'w_input_{i}'].addObjects([dic_windows[f'r_input_{i}']])
        dic_windows[f'w_input_{i}'].setWindowTitle(f"{subject_id} - Input")

        # Display output
        dic_windows[f'a_output_{i}'] = a.toAObject(output_vol)
        dic_windows[f'r_output_{i}'] = a.fusionObjects(objects=[dic_windows[f'a_output_{i}']],
                                                    method='VolumeRenderingFusionMethod')
        # custom palette
        pal = a.createPalette('VR-palette')
        if loss_name in ['bce', 'mse']:
            pal.header()['palette_gradients'] = "1;1#0;1;1;0#0.994872;0#0;0;0.694872;0.244444;1;1"
            minVal=0
            maxVal=0.5
        elif loss_name == 'ce':
            pal.header()['palette_gradients'] = "1;1#0;1;0.292308;0.733333;0.510256;0;0.679487;"+\
            "0.733333#1;0#0;0;0.341026;0.111111;0.507692;0.911111;0.697436;0.111111;1;0"
            minVal=-1.6
            maxVal=0.33

        build_gradient(pal)
        dic_windows[f'r_output_{i}'].setPalette('VR-palette', minVal=minVal,
                                        maxVal=maxVal, absoluteMode=True)
        dic_windows[f'r_output_{i}'].releaseAppRef()
        dic_windows[f'r_output_{i}'].assignReferential(referential1)
        dic_windows[f'w_output_{i}'] = a.createWindow('3D', block=block)
        dic_windows[f'w_output_{i}'].addObjects([dic_windows[f'r_output_{i}']])
        dic_windows[f'w_output_{i}'].setWindowTitle(f"{subject_id} - Output")

    print("Loaded and displayed input/output pairs in Anatomist.")

def main():
    parser = argparse.ArgumentParser(description="Save nii files from npy files.")
    parser.add_argument('-p', '--path', type=str, help="Base folder path to the reconstructions.")
    parser.add_argument('-n', '--nsubjects', type=int, default=4, help="Number of subjects to plot.")
    parser.add_argument('-l', '--lossname', type=str, default='bce', help="Name of the loss used for the decoder.")
    parser.add_argument('-s', '--subjects', type=str, default=None, help="List of subjects you want to plot.")


    args = parser.parse_args()

    recon_path = args.path

    if not recon_path:
        print("No path provided. Please specify with -p.")
        return
    if args.subjects: 
        subjects = args.subjects.split(',')
    else:
        subjects=None

    if os.path.isdir(recon_path):
        plot_ana(recon_path, args.nsubjects, args.lossname, subjects)
    else:
        raise FileNotFoundError(f"Path {recon_path} not found.")


if __name__ == "__main__":
    a = ana.Anatomist()
    nb_columns = 2
    block = a.createWindowsBlock(nb_columns) 
    dic_windows = {}

    main()

    # Keep the GUI application open
    qt_app = Qt.QApplication.instance()
    if qt_app is not None:
        qt_app.exec_()

