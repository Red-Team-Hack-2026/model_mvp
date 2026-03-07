# Find My Force — RF ML Starter (IQ -> class + embedding)

This starter trains a 1D CNN on IQ snapshots stored in one or more HDF5 files.

**Assumptions**
- Each HDF5 dataset key is a string that parses as: (modulation, signal_name, snr_db, sample_idx)
- Each dataset value is a float32 vector of length 256:
  - [0..127] = I
  - [128..255] = Q

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy h5py
```

## Train on one or more HDF5 files
```bash
python train.py --h5 /path/to/RadComDynamic.hdf5 /path/to/Other1.hdf5 /path/to/Other2.hdf5 --out runs/friendly_cnn
```

Outputs in `runs/friendly_cnn/`:
- `best.pt` : model checkpoint
- `meta.json` : label list (modulation + signal_name -> id)
- `centroids.npy` : embedding centroids per friendly class (used for OOD / unknown checks)

## Run inference on a specific HDF5 key
```bash
python infer.py --ckpt runs/friendly_cnn/best.pt --h5 /path/to/RadComDynamic.hdf5 --key "('bpsk', 'Satcom', -10, 0)"
```
To run the full pipeline with the real API:
```bash
python live_feed_client.py | python live_pipeline.py --ckpt runs/friendly_cnn/best.pt | python geolocate.py --receivers receivers.json --path-loss path_loss.json | python track_manager.py
```
To run the full pipeline with mock data:
```bash
python mock_feed.py \
  --mode dataset \
  --h5 RadComDynamic.hdf5 \
  --key "('bpsk', 'Satcom', -10, 42)" \
| python live_pipeline.py --ckpt runs/friendly_cnn/best.pt \
| python geolocate.py --receivers receivers.json --path-loss path_loss.json \
| python track_manager.py