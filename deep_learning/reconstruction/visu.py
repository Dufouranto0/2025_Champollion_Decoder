#visu.py
"""
bv bash

cd 2025_Champollion_Decoder/deep_learning

python3 reconstruction/visu.py \
  -p example \
  -l bce \
  -s sub-1110622,sub-1150302
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

def plot_ana(recon_dir, n_subjects_to_display, loss_name, listsub):#, display_input):

    referential1 = a.createReferential()

    if listsub:
        decoded_files = [os.path.join(recon_dir, f"{sub}_decoded.nii.gz") for sub in listsub]

    else:
        print("No list of subjects given in argument, take 4 random subjects.")
        # Find sub-XXXX_decoded.nii.gz 
        decoded_files = glob.glob(os.path.join(recon_dir, "sub-*_decoded.nii.gz"))

        # Limit to N subjects
        numbers = np.random.choice(len(decoded_files), size=n_subjects_to_display, replace=False)
        decoded_files = [decoded_files[i] for i in numbers]
        print("Decoded files:")
        print([os.path.basename(file) for file in decoded_files])

    for i, decoded_path in enumerate(decoded_files):
        subject_id = os.path.basename(decoded_path).split('_decoded')[0]

        if True: # display_input
            input_path = os.path.join(recon_dir, f"{subject_id}_input.nii.gz")

            if not os.path.isfile(input_path):
                print(f"Missing input for {subject_id}, skipping.")
                #continue
            else:
                # Read input and decoded volumes
                input_vol = aims.read(input_path)

                # Display input
                dic_windows[f'a_input_{i}'] = a.toAObject(input_vol)
                dic_windows[f'r_input_{i}'] = a.fusionObjects(objects=[dic_windows[f'a_input_{i}']],
                                                            method='VolumeRenderingFusionMethod')
                dic_windows[f'r_input_{i}'].releaseAppRef()
                dic_windows[f'r_input_{i}'].assignReferential(referential1)
                dic_windows[f'w_input_{i}'] = a.createWindow('3D', block=block)
                dic_windows[f'w_input_{i}'].addObjects([dic_windows[f'r_input_{i}']])
                dic_windows[f'w_input_{i}'].setWindowTitle(f"{subject_id} - Input")

        # Display decoded
        decoded_vol = aims.read(decoded_path)

        dic_windows[f'a_decoded_{i}'] = a.toAObject(decoded_vol)
        dic_windows[f'r_decoded_{i}'] = a.fusionObjects(objects=[dic_windows[f'a_decoded_{i}']],
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
        dic_windows[f'r_decoded_{i}'].setPalette('VR-palette', minVal=minVal,
                                        maxVal=maxVal, absoluteMode=True)
        dic_windows[f'r_decoded_{i}'].releaseAppRef()
        dic_windows[f'r_decoded_{i}'].assignReferential(referential1)
        dic_windows[f'w_decoded_{i}'] = a.createWindow('3D', block=block)
        dic_windows[f'w_decoded_{i}'].addObjects([dic_windows[f'r_decoded_{i}']])
        dic_windows[f'w_decoded_{i}'].setWindowTitle(f"{subject_id} - decoded")

    print("Loaded and displayed input/decoded pairs in Anatomist.")

def main():
    parser = argparse.ArgumentParser(description="Save nii files from npy files.")
    parser.add_argument('-p', '--path', type=str, help="Base folder path to the reconstructions.")
    parser.add_argument('-n', '--nsubjects', type=int, default=4, help="Number of subjects to plot.")
    parser.add_argument('-l', '--lossname', type=str, default='bce', help="Name of the loss used for the decoder.")
    parser.add_argument('-s', '--subjects', type=str, default=None, help="List of subjects you want to plot.")
    #parser.add_argument('-i', '--displayinput', type=bool, default=True, help="Display encoder input volumes.")


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
        plot_ana(recon_path, args.nsubjects, args.lossname, subjects)#, args.displayinput)
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

