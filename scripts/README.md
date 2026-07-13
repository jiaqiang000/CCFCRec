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
| `run_task4_competitor_pair_m10r4_seed43_fast_uniform_mps_smoke_1epoch.sh` | 本地 Mac MPS（Mac Metal 后端）冒烟测试 launcher（封装启动脚本）。 | M10-R4 competitor pair carrier：real/shuffle/RSP/Acat controls，alpha 默认 0.25，seed43，workers8，fast_uniform，1epoch |
| `run_task4_competitor_pair_m10r4_seed43_fast_uniform_mps_100epoch.sh` | 本地 Mac MPS（Mac Metal 后端）完整快筛 launcher（封装启动脚本）。 | M10-R4 competitor pair carrier：real/shuffle/RSP/Acat controls，alpha 默认 0.25，seed43，workers8，fast_uniform，100epoch |
| `run_task4_post_path_audit_qonly_rescue_seed43_fast_uniform_mps_smoke_1epoch.sh` | 历史旁支 launcher（封装启动脚本），不作为 M11 主线。 | Post path-audit q-only rescue：real/shuffle/RSP/Acat controls，alpha 默认 0.75，seed43，workers8，fast_uniform，1epoch |
| `run_task4_post_path_audit_qonly_rescue_seed43_fast_uniform_mps_100epoch.sh` | 历史旁支 100epoch launcher（封装启动脚本），不作为 M11 主线。 | Post path-audit q-only rescue：real/shuffle/RSP/Acat controls，alpha 默认 0.75，seed43，workers8，fast_uniform，100epoch |
| `watch_task4_post_path_audit_qonly_rescue_logs.sh` | 历史旁支日志查看脚本。 | 配合上面两个 launcher 查看 `launcher_manifest.env`、`master.log` 和各 run 日志 |
| `run_m11_target_competitor_pair_seed43_fast_uniform_mps_smoke_1epoch.sh` | 本地 Mac MPS（Mac Metal 后端）M11 冒烟测试 launcher（封装启动脚本）。 | M11 target-aware competitor-pair loss：real/shuffle/low-RSP/RSP controls，alpha 默认 0.25，seed43，workers8，fast_uniform，1epoch |
| `run_m11_target_competitor_pair_seed43_fast_uniform_mps_100epoch.sh` | 本地 Mac MPS（Mac Metal 后端）M11 正式 single-seed（单随机种子）训练 launcher（封装启动脚本）。 | M11 target-aware competitor-pair loss：real/shuffle/low-RSP/RSP controls，覆盖 baseline 74epoch 峰值窗口，100epoch |
| `watch_m11_target_competitor_pair_logs.sh` | 读取最新 M11 target competitor pair 结果目录并查看日志。 | 配合上面两个 M11 launcher 查看 `launcher_manifest.env`、`master.log` 和各 run 日志 |
| `run_m11r1_full_target_exposure_controls_seed43_fast_uniform_mps_100epoch.sh` | 本地 Mac MPS（Mac Metal 后端）M11-R1 exploratory-only（仅探索）100epoch launcher（封装启动脚本）；不会自动运行。 | original full target（原始完整目标）/popularity-matched control（流行度匹配控制）/low-Acat control（低类别可用性控制），seed43，workers8，fast_uniform，100epoch |
| `watch_m11r1_full_target_exposure_controls_logs.sh` | 读取最新 M11-R1 结果目录并查看日志。 | 配合 M11-R1 100epoch launcher 查看 `launcher_manifest.env`、`master.log` 和各 run 日志 |
| `run_m11r2_seven_experiments_seed43_fast_uniform_mps_100epoch.sh` | 本地 Mac MPS（Mac Metal 后端）M11-R2 七分支统一 launcher（封装启动脚本）；不会自动运行。 | 四个完整 M11 目标性能机制 + 原 M11 方法 + 流行度匹配控制 + 低类别可用性控制；seed43，workers8，fast_uniform，每分支100epoch |
| `watch_m11r2_seven_experiments_logs.sh` | 读取最新 M11-R2 七分支结果并显示总进度。 | 默认输出快照；`FOLLOW=1` 时每 20 秒刷新分支状态、最新 epoch（训练轮次）和主日志 |
| `run_m11r3_four_e4_followups_seed43_fast_uniform_mps_100epoch.sh` | M11-R3 四个 E4 后续机制的统一 launcher（封装启动脚本）；带推荐前特征字段审计，不会自动运行。 | 双路径残差、范数受限残差、训练邻域迁移、目标条件 FiLM（特征线性调制）；seed43，workers8，fast_uniform，每分支100epoch |
| `watch_m11r3_four_e4_followups_logs.sh` | 读取最新 M11-R3 四分支结果并显示总进度。 | 默认输出快照；`FOLLOW=1` 时持续刷新状态、最新 epoch（训练轮次）和主日志 |
| `run_m11r4_four_performance_first_seed43_fast_uniform_mps_100epoch.sh` | M11-R4 四个 performance-first（性能优先）机制的最终 launcher（封装启动脚本）；固定 MPS（苹果图形处理器）与100epoch（100训练轮次），剔除测试行并拒绝验证/测试结果列。 | 目标保护双专家、连续语义融合、目标关系对齐、连续难例聚焦优化；seed43，workers8，fast_uniform，每分支100epoch |
| `watch_m11r4_four_performance_first_logs.sh` | 读取最新 M11-R4 四分支结果目录并显示状态与日志尾部。 | 配合 M11-R4 最终 launcher（封装启动脚本）查看 `status.tsv`、`master.log` 和四个分支日志 |
| `run_cicpr1_six_access_methods_seed43_fast_uniform_mps_100epoch.sh` | CICP-R1（类别增量协同可预测性第一轮）六机制统一 launcher（封装启动脚本）；正式运行强制 MPS（苹果图形处理器）、fast_uniform（快速均匀负采样）和100epoch（100训练轮次）。 | 唯一E4式残差、模态路由、类别专家混合、协同对齐课程、类别反事实间隔、自适应类别注意力；seed43，workers8，每分支100epoch |
| `watch_cicpr1_six_access_methods_logs.sh` | 读取最新 CICP-R1（类别增量协同可预测性第一轮）六分支结果并显示状态、完成数和最新训练轮次。 | 默认输出一次快照；使用 `--follow` 时每30秒持续刷新 |
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
