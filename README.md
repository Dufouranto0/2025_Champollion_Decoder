```markdown
# Champollion-Decoder
```

```bash
git clone https://github.com/Dufouranto0/2025_Champollion_Decoder.git
cd 2025_Champollion_Decoder
python -m venv decodervenv
. decodervenv/bin/activate
pip install -r requirements.txt
```

## Training a decoder for a specific model

To train a decoder for a specific model, modify the `model_to_decode_path` field in the config file to point to the **directory of the model** you want to decode.  

Also make sure that by concatenating `model_to_decode_path` and `train_csv`, you get the path to the **embeddings corresponding to the subjects used during the training of the encoder**.  

The `val_test_csv` corresponds to embeddings containing subjects used for **validation** in the encoder training.

Then run:

```bash
cd deep_learning
python3 train.py
```

---

## Saving NIfTI files from decoder outputs

If you want to save NIfTI files from the NumPy outputs of the decoder (for now only the first two batches for visualization), use `save_nii.py` inside a BrainVisa environment (or any environment that contains `aims`):

```bash
bv bash
cd 2025_Champollion_Decoder/deep_learning
python3 reconstruction/save_nii.py -p example
```

---

## Comparing encoder input with decoder output

To compare the initial encoder input with the decoder output, you also need a BrainVisa environment:

```bash
bv bash
cd 2025_Champollion_Decoder/deep_learning
python3 reconstruction/visu.py \
  -p example \
  -l bce \
  -s sub-1110622_input.nii.gz,sub-1150302_input.nii.gz
```

## Decoder Architecture

![Decoder Architecture](figures/decoder_architecture.png)

## Example Visualization

![Decoder Output Example](figures/SOr_left.png)
