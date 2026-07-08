# CCFCRec Scripts

本目录只放可复用或需要沉淀的训练启动脚本。一次性命令不应只停留在 shell history（终端历史）里。

| 脚本 | 定位 | 对应实验 |
|---|---|---|
| `train_amazon_vg_cuda.sh` | 通用 Amazon-VG 训练入口包装脚本；本地 MPS 和服务器 CUDA 都可以通过环境变量切换。 | 多个实验复用 |
| `run_task4_acat_v3_weight_controls_m1_m2_m6_m3_seed43_fast_uniform_mps_100epoch.sh` | 本地 Mac MPS（Mac Metal 后端）实验专用 launcher（封装启动脚本）。 | Task4 Acat_v3 minimal weight controls：M1/M2/M6/M3，seed43，workers8，fast_uniform，100epoch |
| `run_task4_acat_v3_pairmargin_candidates_m4a_m4b_m4c_seed43_fast_uniform_mps_smoke_1epoch.sh` | 本地 Mac MPS（Mac Metal 后端）冒烟测试 launcher（封装启动脚本）。 | Task4-revise Acat_v3 pair-margin candidates：M4a/M4b/M4c，seed43，workers8，fast_uniform，1epoch |
| `run_task4_acat_v3_pairmargin_candidates_m4a_m4b_m4c_seed43_fast_uniform_mps_100epoch.sh` | 本地 Mac MPS（Mac Metal 后端）完整筛选 launcher（封装启动脚本）。 | Task4-revise Acat_v3 pair-margin candidates：M4a/M4b/M4c，seed43，workers8，fast_uniform，100epoch |
| `run_task4_highdetail_acat_trainhard_carriers_m7_seed43_fast_uniform_mps_smoke_1epoch.sh` | 本地 Mac MPS（Mac Metal 后端）冒烟测试 launcher（封装启动脚本）。 | Task4-revise-2 high-detail high-Acat trainhard carriers：M7a/M7s/M7b/M7ps，seed43，workers8，fast_uniform，1epoch |
| `run_task4_highdetail_acat_trainhard_carriers_m7_seed43_fast_uniform_mps_100epoch.sh` | 本地 Mac MPS（Mac Metal 后端）完整筛选 launcher（封装启动脚本）。 | Task4-revise-2 high-detail high-Acat trainhard carriers：M7a/M7s/M7b/M7ps，seed43，workers8，fast_uniform，100epoch |
| `run_task4_highdetail_m7a_surface_ablation_seed43_fast_uniform_mps_smoke_1epoch.sh` | 本地 Mac MPS（Mac Metal 后端）冒烟测试 launcher（封装启动脚本）。 | Task4-revise-3 M7a high-detail carrier surface ablation：M8q/M8qs/M8s/M8ss，seed43，workers8，fast_uniform，1epoch |
| `run_task4_highdetail_m7a_surface_ablation_seed43_fast_uniform_mps_100epoch.sh` | 本地 Mac MPS（Mac Metal 后端）完整筛选 launcher（封装启动脚本）。 | Task4-revise-3 M7a high-detail carrier surface ablation：M8q/M8qs/M8s/M8ss，seed43，workers8，fast_uniform，100epoch |
| `run_task4_highdetail_qonly_alpha_sweep_m9_seed43_fast_uniform_mps_smoke_1epoch.sh` | 本地 Mac MPS（Mac Metal 后端）冒烟测试 launcher（封装启动脚本）。 | Task4-revise-4 M9 q-only alpha sweep：alpha 0.75/1.0 各自 real/shuffle，seed43，workers8，fast_uniform，1epoch |
| `run_task4_highdetail_qonly_alpha_sweep_m9_seed43_fast_uniform_mps_100epoch.sh` | 本地 Mac MPS（Mac Metal 后端）完整筛选 launcher（封装启动脚本）。 | Task4-revise-4 M9 q-only alpha sweep：alpha 0.75/1.0 各自 real/shuffle，seed43，workers8，fast_uniform，100epoch |
| `run_task4_full_100epoch_mps.sh` | 兼容旧命令的 wrapper（封装转发脚本）。 | 转发到上面的 Task4 实验专用 launcher |

实验专用 launcher 应在结果目录写入 `launcher_manifest.env`，至少记录：

```text
EXPERIMENT_ID
EXPERIMENT_DESIGN_NOTE
LAUNCHER_SCRIPT
RESULT_ROOT
TASK4_PROFILE 或对应输入 profile
METHOD_VARIANTS
SEED / NUM_WORKERS / BATCH_SIZE / NEGATIVE_SAMPLING_MODE / CCFCREC_DEVICE / EPOCH
```

后续排查优先顺序：

```text
结果目录/run_manifest.json 或 launcher_manifest.env
实验记录/ 对应设计 MD 和路线判断 MD
scripts/README.md
具体 launcher 脚本
```
