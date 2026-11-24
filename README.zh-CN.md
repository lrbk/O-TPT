# O-TPT 复现与改进实验仓库 —— 文件说明

> 本仓库是对 CVPR 2025 论文 **[O-TPT](https://arxiv.org/abs/2503.12096)**（Orthogonality Constraints for Calibrating Test-time Prompt Tuning in Vision-Language Models）的复现，并在原版基础上派生了多个测试时提示调优（TPT）校准改进变体。论文原版说明见 [README.md](README.md)（英文）。

## 目录

1. [仓库结构总览](#1-仓库结构总览)
2. [顶层入口文件](#2-顶层入口文件)
3. [clip/ 模型模块](#3-clip-模型模块)
4. [data/ 数据模块](#4-data-数据模块)
5. [utils/ 工具模块](#5-utils-工具模块)
6. [scripts/ 脚本](#6-scripts-脚本)
7. [log/ 实验产物](#7-log-实验产物)
8. [根目录其他文件](#8-根目录其他文件)
9. [快速开始](#9-快速开始)
10. [不在仓库内的内容](#10-不在仓库内的内容)

---

## 1. 仓库结构总览

```text
O-TPT-sync/
├── clip/              # CLIP 编码器、PromptLearner、各变体模型定义
├── data/              # 数据集加载、增强、提示模板、few-shot 划分
├── utils/             # 校准指标（ECE 等）与通用工具
├── scripts/           # 原版实验脚本、复现脚本、批量/网格实验启动器
├── log/               # 精选实验结果（csv/tsv、summary.csv、manifest）
├── otpt_classification*.py   # 7 个实验入口（原版 + 6 个改进派生版）
├── OTPT_Improvement_Directions.md  # 改进方向调研笔记
├── environment.yml    # conda 环境定义
├── README.md          # 论文原版英文 README
├── README.zh-CN.md    # 本文件：项目文件说明
├── LICENSE
├── otpt-results.png / radar_plot_3_.jpg  # 论文结果图
└── .gitignore
```

## 2. 顶层入口文件

每个入口都是独立可运行的分类实验主程序，负责参数解析、数据加载、逐样本测试时自适应、指标计算与 CSV 落盘。所有派生版均以原版为基线，**不修改原版文件**。

| 文件 | 对应的 clip 模型 | 定位与关键参数 |
|---|---|---|
| [otpt_classification.py](otpt_classification.py) | `custom_clip_iptp_bas` | **原版 O-TPT 基线**。支持 `--run_type {baseline, tpt, tpt_otpt, tpt_ts}`、`--lambda_term`（C-TPT 正交权重）、`--selection_p`、`--n_ctx` 等 |
| [otpt_classification_trust_region.py](otpt_classification_trust_region.py) | `custom_clip_iptp_bas` | **信任域变体（最小派生）**。新增 `--use_trust_region`、`--lam_trust_region`、`--trust_region_mode {ctx, text, both}`，分别约束 prompt context / 文本特征相对 zero-shot 锚点的漂移；不传开关或 λ=0 时执行路径与原版完全一致 |
| [otpt_classification_nc_sem.py](otpt_classification_nc_sem.py) | `custom_clip_otpt_sem` | **集成度最高的实验入口**，通过开关切换多条改进路线：`--use_nc_sem`（NC-TPT + 语义 Gram 正则，含 `--sem_alpha`、`--sem_prior_path`）、`--use_mm_ortho`（跨模态正交）、`--use_cache_ortho`（高置信原型缓存）、`--use_dist_align`、`--use_proda_ortho`、`--use_adv_ortho`、`--use_entropy_target`、`--use_logitnorm_tpt`、`--use_trust_region`；另含 `--max_test_samples` 子集评测和 `--temperature_grid` 温度网格 |
| [otpt_classification_adaptive_lambda.py](otpt_classification_adaptive_lambda.py) | `custom_clip_adaptive_lambda` | **自适应 λ 变体**，`--adaptive_lambda` 按样本不确定度动态缩放正交损失权重 |
| [otpt_classification_dual_guided_v2.py](otpt_classification_dual_guided_v2.py) | `custom_clip_dual_guided_v2` | **视觉-文本双引导 v2**，引入可学习残差门控，参数 `--gate_init`（默认 0.1）、`--gate_temp`（门控 sigmoid 温度，默认 2.0） |
| [otpt_classification_safe_anchor.py](otpt_classification_safe_anchor.py) | `custom_clip_iptp_bas` | **安全锚点变体**，复用原版模型、只在入口层增加锚点 KL 约束逻辑，同样带 `--gate_init/--gate_temp` 门控参数 |
| [otpt_classification_psr_im.py](otpt_classification_psr_im.py) | `custom_clip_psr_im` | **PSR + 信息最大化等多策略入口**：`--psr_T` 伪目标锐化温度、`--lam_im` 边缘熵权重、`--use_etf`/`--lam_etf` Neural Collapse ETF 正则、`--use_anchor_prior` 锚点 KL 信任域（`--lam_anchor`、`--anchor_topm`、`--anchor_conf_gate`）、`--lam_pull` prompt 残差回拉 |

### 入口通用参数（各入口基本一致）

- `data`（位置参数）：数据集根目录；`--test_sets`：数据集代号（如 Flower102/Pets/DTD）
- `-a/--arch`：CLIP 骨干（默认 RN50，实验中常用 ViT-B/16）；`--resolution` 输入分辨率
- `--tpt`：开启测试时提示调优；`--tta_steps` 自适应步数；`--n_ctx` 可学习 token 数
- `--seed`：随机种子；`--csv_log`：结果 CSV 输出路径；`--gpu` 指定 GPU

## 3. clip/ 模型模块

| 文件 | 说明 |
|---|---|
| [clip.py](clip/clip.py) | 官方 CLIP 封装：`load()` 加载权重、`tokenize()` 文本分词、模型下载与预处理 |
| [model.py](clip/model.py) | CLIP 网络结构：ResNet（`ModifiedResNet`）、Transformer 残差块、`QuickGELU` 等 |
| [simple_tokenizer.py](clip/simple_tokenizer.py) + `bpe_simple_vocab_16e6.txt.gz` | BPE 分词器与词表 |
| [custom_clip.py](clip/custom_clip.py) | 基础 TPT 模型：`ClipImageEncoder`、`TextEncoder`、`PromptLearner`、`ClipTestTimeTuning`、`get_coop()` 工厂函数，是所有定制模型的骨架 |
| [custom_clip_iptp_bas.py](clip/custom_clip_iptp_bas.py) | **原版 O-TPT 模型**（Householder/Gram 文本正交约束），被原版、trust-region、safe-anchor 三个入口共用 |
| [custom_clip_otpt_sem.py](clip/custom_clip_otpt_sem.py) | 语义感知正交模型，服务 nc_sem 入口 |
| [custom_clip_adaptive_lambda.py](clip/custom_clip_adaptive_lambda.py) | 自适应 λ 模型 |
| [custom_clip_dual_guided_v2.py](clip/custom_clip_dual_guided_v2.py) / [custom_clip_safe_anchor.py](clip/custom_clip_safe_anchor.py) | 双引导 v2 / 安全锚点模型（600 行级，含门控逻辑，`get_coop` 带 `gate_init/gate_temp`） |
| [custom_clip_psr_im.py](clip/custom_clip_psr_im.py) | PSR-IM 模型，额外提供 `build_nc_classifier()` 构造 NC 分类器 |
| [cocoop.py](clip/cocoop.py) | CoCoOp 提示学习器（`CoCoOpPromptLearner`、`get_cocoop`），用于提示初始化 |

## 4. data/ 数据模块

| 文件 | 说明 |
|---|---|
| [datautils.py](data/datautils.py) | 核心数据管道：`build_dataset()` 按代号构建数据集、`AugMixAugmenter` 测试时增强、`get_preaugment()` |
| [fewshot_datasets.py](data/fewshot_datasets.py) | 各 fine-grained 数据集类（Aircraft 等）与 `build_fewshot_dataset()`、`BaseJsonDataset` |
| [augmix_ops.py](data/augmix_ops.py) | AugMix 图像增强算子集（自动对比度、均衡化等） |
| [imagnet_prompts.py](data/imagnet_prompts.py) | ImageNet 类名提示模板 |
| [cls_to_names.py](data/cls_to_names.py) | 类别 ID → 类名映射 |
| [imagenet_variants.py](data/imagenet_variants.py) | ImageNet-A/R/V2/Sketch 变体数据集辅助 |
| [hoi_dataset.py](data/hoi_dataset.py) | Bongard HOI 数据集加载 |
| [covid_prompts.py](data/covid_prompts.py) | COVID 数据集提示（占位模块） |
| `data_splits/` | few-shot 划分 JSON（OxfordFlowers、OxfordPets、DescribableTextures） |

## 5. utils/ 工具模块

| 文件 | 说明 |
|---|---|
| [metrics.py](utils/metrics.py) | 校准指标：`expected_calibration_error`（ECE）、`maximum_calibration_error`（MCE）、`average_calibration_error`（ACE）、L2 error，以及 logits 级评测函数 |
| [tools.py](utils/tools.py) | `set_random_seed()`、`AverageMeter/ProgressMeter`、`accuracy()`、`CELoss/SCELoss`、`load_model_weight()` 等通用工具 |

## 6. scripts/ 脚本

按用途分四组：

### 6.1 论文原版实验（`test_*.sh`）

| 脚本 | 用途 |
|---|---|
| [test_baseline.sh](scripts/test_baseline.sh) | zero-shot CLIP 基线 |
| [test_tpt_fg.sh](scripts/test_tpt_fg.sh) / [test_tpt_ds.sh](scripts/test_tpt_ds.sh) | TPT：fine-grained / 分布漂移数据集 |
| [test_tpt_otpt_fg.sh](scripts/test_tpt_otpt_fg.sh) / [test_tpt_otpt_ds.sh](scripts/test_tpt_otpt_ds.sh) | O-TPT：fine-grained / 分布漂移数据集 |

### 6.2 本地复现脚本（`reproduce_*_local.sh`）

覆盖原版及各派生入口：`reproduce_otpt_local.sh`、`reproduce_otpt_nc_sem_original_local.sh`、`reproduce_otpt_adaptive_lambda_local.sh`、`reproduce_otpt_nc_local.sh`、`reproduce_otpt_psr_im_local.sh`、`reproduce_dual_guided_v2_local.sh`。

### 6.3 批量 / 网格实验（`run_*.sh`）

| 脚本 | 用途 |
|---|---|
| [run_trust_region_grid.sh](scripts/run_trust_region_grid.sh) | trust-region 网格：mode ∈ {ctx, text, both} × λ ∈ {0, 0.001, 0.003, 0.01, 0.03, 0.1}，产出 `summary.csv` |
| [run_direction_smoke_matrix.sh](scripts/run_direction_smoke_matrix.sh) | 多条改进方向的子集冒烟矩阵 |
| [run_otpt_fullseed_fg.sh](scripts/run_otpt_fullseed_fg.sh) | fine-grained 数据集三种子全量跑 |
| [run_dual_guided_v2.sh](scripts/run_dual_guided_v2.sh) / [run_official_safe_anchor.sh](scripts/run_official_safe_anchor.sh) / [run_psr_im_single_seed.sh](scripts/run_psr_im_single_seed.sh) / [run_otpt_nc_single_seed.sh](scripts/run_otpt_nc_single_seed.sh) | 对应变体的单种子运行器 |

### 6.4 远程 Worker 启动器（`launch_*.sh`）与状态检查

`launch_otpt_worker.sh`、`launch_otpt_fullseed_worker.sh`、`launch_otpt_adaptive_lambda_worker.sh`、`launch_dual_guided_v2.sh`、`launch_psr_*.sh`、`launch_nc_single_worker.sh` 用于在 GPU worker 上后台拉起实验；[check_otpt_worker.sh](scripts/check_otpt_worker.sh) 用于检查 worker 运行状态。

## 7. log/ 实验产物

只保留轻量文本结果（整个目录约 2 MB），stdout 全量日志、pid、nohup 等运行时文件未纳入仓库。

**目录约定**：每个批量实验一个时间戳目录，内部固定为：

- `csv/`：每个配置一份指标文件（Acc / Top5 / ECE 等）
- `manifest.tsv`：配置清单（命令、状态、输出路径）
- `summary.csv`：网格汇总（网格类实验才有）
- 根目录零散的 `*.csv` / `*.log`：单次实验的直接产物

**主要实验目录**：

| 目录 / 文件 | 内容 |
|---|---|
| `trust_region_grid_*_20260813_180617/` | trust-region 完整网格（18 配置全 OK），最佳 ECE **3.43**（ctx, λ=0.001），优于论文 O-TPT Flower102 的 3.87 |
| `trust_region_grid_*_20260812_150403/`、`trust_region_wrapper_fix_check_*/` | 早前网格与 wrapper 修复验证 |
| `unified_protocol_flower102_*/` | 统一协议下 8 种方法（adv/cache/dist/mm/nc/proda/sem ortho、entropy、logitnorm）全量对比 + 温度网格 |
| `direction_smoke_after_full_*/` | 各改进方向子集冒烟结果 |
| `nc_sem_original_full_*/`、`flower102_remaining_methods_full_*/`、`pets_methods_full_*/` | nc_sem 及各 ortho 方向全量结果 |
| `logitnorm_otpt_sanity_*/` | LogitNorm λ×τ 网格 sanity check |
| `fullseed_*/` | DTD/Flower102/Pets × seed 0/1/2 的完整结果与 `command.sh` |
| `reproduce_raw_otpt_flower102_*/` | 原版 O-TPT 三种子复现 |
| 其余 `reproduce_*.csv`、`a3_*`、`a5_*`、`a6_*`、`psr_*`、`entropy_*` | 历次单方法实验留痕 |

## 8. 根目录其他文件

| 文件 | 说明 |
|---|---|
| [OTPT_Improvement_Directions.md](OTPT_Improvement_Directions.md) | 改进方向调研：O-TPT 三个可攻击面，以及多模态正交、Cache 增强、语义感知正交、分布对齐等方向的动机与实施方案 |
| [environment.yml](environment.yml) | conda 环境 `otpt` 的依赖定义 |
| [otpt-results.png](otpt-results.png)、[radar_plot_3_.jpg](radar_plot_3_.jpg) | 论文结果配图 |
| [LICENSE](LICENSE) | 原仓库许可证 |
| [.gitignore](.gitignore) | 忽略权重缓存、`dataset/` 符号链接、`__pycache__`、pid/nohup/stdout 等 |

## 9. 快速开始

```bash
# 1. 环境
conda env create -f environment.yml
conda activate otpt

# 2. 按 TPT 官方约定准备数据集，并把根目录传给入口的位置参数

# 3. 跑一次原版 O-TPT（fine-grained）
bash scripts/test_tpt_otpt_fg.sh Flower102

# 4. 跑 trust-region 网格（产出 log/trust_region_grid_*/summary.csv）
bash scripts/run_trust_region_grid.sh
```

## 10. 不在仓库内的内容

以下内容受体积或机器相关性限制，未纳入版本管理（见 `.gitignore`）：

- **CLIP 权重** `ViT-B-16.pt`（335 MB）：首次运行时由 `clip.load()` 自动下载
- **数据集**：原 `dataset/` 下均为指向本机/挂载路径的符号链接，需按 [TPT 官方仓库](https://github.com/azshue/TPT) 自行准备
- `__pycache__/`、`*.pyc`、pid/nohup/launcher 文件、`log/**/stdout/` 全量标准输出
