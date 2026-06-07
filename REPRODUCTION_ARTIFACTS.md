# CCFCRec Reproduction Artifacts

This repository does not use ClearML or a full experiment manager. Training writes a small set of files under each dataset directory's `result/` folder, while shell logs and environment metadata must be captured by the run command.

## Amazon VG

### Data Location

Expected server data directory:

```text
/root/CCFCRec/Amazon VG/data/
```

Required files:

```text
asin.csv
asin_int_category.pkl
img_feature.npy
test_rating.csv
train_rating.csv
train_withneg_rating.csv
validate_rating.csv
```

### Smoke Test

Use this to confirm data, code, CUDA, and the multi-worker data loader work:

```bash
cd "/root/CCFCRec/Amazon VG"
export CCFCREC_CUDA_DEVICE=0
/usr/bin/time -v python model.py --epoch 1 --save_batch_time 999999999 --num_workers 8 --pin_memory --persistent_workers 2>&1 | tee smoke_amazon_vg_workers8.log
```

The smoke command does not trigger validation or checkpoint saving because `save_batch_time` is intentionally larger than the number of batches.

### Full Run

Amazon VG has 326 batches per epoch with the default `batch_size=1024`. `save_batch_time=300` triggers validation and checkpoint saving roughly once per epoch.

```bash
cd "/root/CCFCRec/Amazon VG"
export CCFCREC_CUDA_DEVICE=0
/usr/bin/time -v python model.py --epoch 100 --save_batch_time 300 --num_workers 8 --pin_memory --persistent_workers 2>&1 | tee run_amazon_vg_workers8.log
```

### Generated Files

Training creates a timestamped result directory:

```text
/root/CCFCRec/Amazon VG/result/YYYY-MM-DD_HH_MM_SS/
```

Expected contents:

```text
readme.txt       # hyperparameters and training start time
result.csv       # validation metrics written every save_batch_time batches
save_dict.pkl    # item/user/category mappings needed by test.py
1.pt, 2.pt, ...  # checkpoints saved every save_batch_time batches
```

### Must Download

Download the run log:

```text
/root/CCFCRec/Amazon VG/run_amazon_vg_workers8.log
```

Download the complete latest result directory:

```bash
ls -td "/root/CCFCRec/Amazon VG/result/"* | head -1
```

At minimum, preserve these files from that directory:

```text
readme.txt
result.csv
save_dict.pkl
*.pt
```

### Recommended Metadata

Before or after the full run, write a small metadata file:

```bash
cd /root/CCFCRec
{
  echo "commit: $(git rev-parse HEAD)"
  echo "date: $(date)"
  echo "cmd: python model.py --epoch 100 --save_batch_time 300 --num_workers 8 --pin_memory --persistent_workers"
  python -c "import torch, numpy, pandas, tqdm; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('tqdm', tqdm.__version__)"
  nvidia-smi
} > "/root/CCFCRec/Amazon VG/run_meta_amazon_vg_workers8.txt"
```

Download this file too:

```text
/root/CCFCRec/Amazon VG/run_meta_amazon_vg_workers8.txt
```

## ML-20M

### Data Location

Expected server data directory:

```text
/root/CCFCRec/ML-20M/data/
```

Required files:

```text
img_feature.csv
movies_onehot.csv
self_contrast_48.csv
test_rating.csv
train_rating.csv
user_positive_movie_48.csv
validate_rating.csv
```

Optional cache:

```text
/root/CCFCRec/ML-20M/pkl/user_pn_dict.pkl
```

### Smoke Test

```bash
cd "/root/CCFCRec/ML-20M"
export CCFCREC_CUDA_DEVICE=0
/usr/bin/time -v python model.py --epoch 1 --save_batch_time 999999999 --num_workers 8 --pin_memory --persistent_workers 2>&1 | tee smoke_ml20m_workers8.log
```

The smoke command does not trigger validation or checkpoint saving.

### Full Run

ML-20M has 13,606 batches per epoch with the default `batch_size=1024`. The original default `save_batch_time=3000` triggers several validation/checkpoint events per epoch.

```bash
cd "/root/CCFCRec/ML-20M"
export CCFCREC_CUDA_DEVICE=0
/usr/bin/time -v python model.py --epoch 10 --save_batch_time 3000 --num_workers 8 --pin_memory --persistent_workers 2>&1 | tee run_ml20m_workers8.log
```

### Generated Files

Training creates:

```text
/root/CCFCRec/ML-20M/result/YYYY-MM-DD_HH:MM:SS/
```

Expected contents:

```text
readme.txt
result.csv
epoch_0batch_3000.pt
epoch_0batch_6000.pt
...
```

### Must Download

Download:

```text
/root/CCFCRec/ML-20M/run_ml20m_workers8.log
/root/CCFCRec/ML-20M/result/<timestamp>/readme.txt
/root/CCFCRec/ML-20M/result/<timestamp>/result.csv
/root/CCFCRec/ML-20M/result/<timestamp>/*.pt
```

Prefer downloading the complete latest result directory:

```bash
ls -td "/root/CCFCRec/ML-20M/result/"* | head -1
```

### Recommended Metadata

```bash
cd /root/CCFCRec
{
  echo "commit: $(git rev-parse HEAD)"
  echo "date: $(date)"
  echo "cmd: python model.py --epoch 10 --save_batch_time 3000 --num_workers 8 --pin_memory --persistent_workers"
  python -c "import torch, numpy, pandas, tqdm; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('tqdm', tqdm.__version__)"
  nvidia-smi
} > "/root/CCFCRec/ML-20M/run_meta_ml20m_workers8.txt"
```

Download:

```text
/root/CCFCRec/ML-20M/run_meta_ml20m_workers8.txt
```

## Notes

- Keep code and final artifacts under `/root/CCFCRec` if they need to survive instance restarts.
- Use `/hy-tmp` only for temporary downloads, archives, and extraction work.
- Do not keep large archives under `/root` after extraction if disk space is tight.
- The `result/` directories are ignored by Git and must be downloaded manually.
