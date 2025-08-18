import anatomist.api as ana
from soma.qt_gui.qt_backend import Qt
from soma import aims
import os
import glob

# Start Anatomist
a = ana.Anatomist()

# Setup: how many pairs to display
n_subjects_to_display = 4
nb_columns = 2
block = a.createWindowsBlock(nb_columns)  # Create a layout block with 2 columns
dic_windows = {}

# Folder containing .nii.gz volumes
recon_dir = "runs/6_test_bce_5e-4/reconstructions_epoch2"
referential1 = a.createReferential()

# Find subject_XXXX_input.nii.gz and match with output
input_files = sorted(glob.glob(os.path.join(recon_dir, "subject_*_input.nii.gz")))

# Limit to first N subjects
input_files = input_files[:n_subjects_to_display]

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
    dic_windows[f'r_output_{i}'].releaseAppRef()
    dic_windows[f'r_output_{i}'].assignReferential(referential1)
    dic_windows[f'w_output_{i}'] = a.createWindow('3D', block=block)
    dic_windows[f'w_output_{i}'].addObjects([dic_windows[f'r_output_{i}']])
    dic_windows[f'w_output_{i}'].setWindowTitle(f"{subject_id} - Output")

print("Loaded and displayed input/output pairs in Anatomist.")

# Keep the GUI application open
qt_app = Qt.QApplication.instance()
if qt_app is not None:
    qt_app.exec_()