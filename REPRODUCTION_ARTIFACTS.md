# CCFCRec 复现实验产物清单

这个仓库没有使用 ClearML，也没有完整的实验管理器。训练脚本只会在每个数据集目录下的 `result/` 里保存少量结果文件；命令行日志、环境信息、GPU 信息需要我们用 shell 命令额外记录。

## Amazon VG

### 数据位置

服务器上期望的数据目录：

```text
/root/CCFCRec/Amazon VG/data/
```

必须有这些文件：

```text
asin.csv
asin_int_category.pkl
img_feature.npy
test_rating.csv
train_rating.csv
train_withneg_rating.csv
validate_rating.csv
```

### 连通性测试

先用这个命令确认数据、代码、CUDA、多 worker DataLoader 都能正常跑：

```bash
cd "/root/CCFCRec/Amazon VG"
export CCFCREC_CUDA_DEVICE=0
/usr/bin/time -v python model.py --epoch 1 --save_batch_time 999999999 --num_workers 32 --pin_memory --persistent_workers 2>&1 | tee smoke_amazon_vg_workers32.log
```

这里的 `save_batch_time` 故意设得大于总 batch 数，所以这个 smoke test 不会触发验证，也不会保存 checkpoint。它只用于确认训练入口是否能跑起来，以及估算速度和资源占用。

### 正式训练

Amazon VG 默认 `batch_size=1024` 时，每个 epoch 有 326 个 batch。`save_batch_time=300` 基本会在每个 epoch 末尾触发一次验证和 checkpoint 保存。当前服务器上 `num_workers=32` 的 smoke test 速度明显优于 `8/16`，所以 Amazon VG 正式训练优先使用 32 workers。

```bash
cd "/root/CCFCRec/Amazon VG"
export CCFCREC_CUDA_DEVICE=0
/usr/bin/time -v python model.py --epoch 100 --save_batch_time 300 --num_workers 32 --pin_memory --persistent_workers 2>&1 | tee run_amazon_vg_workers32.log
```

### 训练生成的文件

训练会创建一个带时间戳的结果目录：

```text
/root/CCFCRec/Amazon VG/result/YYYY-MM-DD_HH_MM_SS/
```

目录里通常会有：

```text
readme.txt       # 超参数、保存目录、训练开始时间
result.csv       # 每次验证写入的 loss、HR、NDCG
save_dict.pkl    # test.py 需要用到的 user/item/category 映射
1.pt, 2.pt, ...  # 每次触发 save_batch_time 保存的 checkpoint
```

### 必须下载

训练日志：

```text
/root/CCFCRec/Amazon VG/run_amazon_vg_workers32.log
```

最新结果目录可以这样查：

```bash
ls -td "/root/CCFCRec/Amazon VG/result/"* | head -1
```

建议直接下载整个最新结果目录。至少要保留这些文件：

```text
readme.txt
result.csv
save_dict.pkl
*.pt
```

### 建议额外记录的环境信息

正式训练前或训练后，建议写一份元信息文件，方便后面追踪复现条件：

```bash
cd /root/CCFCRec
{
  echo "commit: $(git rev-parse HEAD)"
  echo "date: $(date)"
  echo "cmd: python model.py --epoch 100 --save_batch_time 300 --num_workers 32 --pin_memory --persistent_workers"
  python -c "import torch, numpy, pandas, tqdm; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('tqdm', tqdm.__version__)"
  nvidia-smi
} > "/root/CCFCRec/Amazon VG/run_meta_amazon_vg_workers32.txt"
```

这个文件也要下载：

```text
/root/CCFCRec/Amazon VG/run_meta_amazon_vg_workers32.txt
```

## ML-20M

### 数据位置

服务器上期望的数据目录：

```text
/root/CCFCRec/ML-20M/data/
```

必须有这些文件：

```text
img_feature.csv
movies_onehot.csv
self_contrast_48.csv
test_rating.csv
train_rating.csv
user_positive_movie_48.csv
validate_rating.csv
```

可选缓存文件：

```text
/root/CCFCRec/ML-20M/pkl/user_pn_dict.pkl
```

这个缓存很小，可以省掉第一次构建用户正样本索引的时间。

### 连通性测试

```bash
cd "/root/CCFCRec/ML-20M"
export CCFCREC_CUDA_DEVICE=0
/usr/bin/time -v python model.py --epoch 1 --save_batch_time 999999999 --num_workers 8 --pin_memory --persistent_workers 2>&1 | tee smoke_ml20m_workers8.log
```

这个 smoke test 不会触发验证，也不会保存 checkpoint。

### 正式训练

ML-20M 默认 `batch_size=1024` 时，每个 epoch 有 13,606 个 batch。原始默认 `save_batch_time=3000`，每个 epoch 会触发多次验证和 checkpoint 保存。

```bash
cd "/root/CCFCRec/ML-20M"
export CCFCREC_CUDA_DEVICE=0
/usr/bin/time -v python model.py --epoch 10 --save_batch_time 3000 --num_workers 8 --pin_memory --persistent_workers 2>&1 | tee run_ml20m_workers8.log
```

### 训练生成的文件

训练会创建：

```text
/root/CCFCRec/ML-20M/result/YYYY-MM-DD_HH:MM:SS/
```

目录里通常会有：

```text
readme.txt
result.csv
epoch_0batch_3000.pt
epoch_0batch_6000.pt
...
```

### 必须下载

```text
/root/CCFCRec/ML-20M/run_ml20m_workers8.log
/root/CCFCRec/ML-20M/result/<timestamp>/readme.txt
/root/CCFCRec/ML-20M/result/<timestamp>/result.csv
/root/CCFCRec/ML-20M/result/<timestamp>/*.pt
```

更建议直接下载最新结果目录：

```bash
ls -td "/root/CCFCRec/ML-20M/result/"* | head -1
```

### 建议额外记录的环境信息

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

这个文件也要下载：

```text
/root/CCFCRec/ML-20M/run_meta_ml20m_workers8.txt
```

## 注意事项

- 需要长期保留的代码、日志、结果目录放在 `/root/CCFCRec`。
- `/hy-tmp` 只适合放临时下载的压缩包、临时解压目录和中转文件。
- 根目录空间只有 30GB 左右，不要在 `/root` 长期保留大压缩包。
- `result/` 已经被 `.gitignore` 忽略，不会通过 Git 同步，必须手动下载。
