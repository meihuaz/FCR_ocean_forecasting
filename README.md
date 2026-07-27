# FCR_ocean_forecasting

Feature-Consistency Robust Training (FCR) for robust AI-based ocean forecasting.

本目录汇集当前论文中方法、实验和主要结果图所涉及的源代码。文件从原工程复制而来，原文件未移动或删除；目录层级保持不变，以保留现有 Python 导入关系。

## 内容对应关系

### 训练与评估

- `train_nw_pacific.py`：标准监督预训练。
- `train_online_continue_simsiam.py`：FCR/PGD/特征一致性微调及消融配置。
- `test_nw_pacific.py`：标准模型单步评估。
- `test_online_continue_simsiam.py`：FCR 模型的 clean/PGD 评估，可通过不同 `--pgd_epsilon` 运行扰动幅度实验。
- `adversarial_generation.py`：受 `L_inf` 约束的 PGD 最坏情形扰动生成。

### 真实初始化误差与论文图

- `data_process/evaluate_analysis_init_reanalysis_truth_simsiam.py`：GLORYS 分析场—再分析场成对评估，并可生成论文真实初始化误差案例图（图 3）。
- `data_process/compare_glory_analysis_reanalysis.py`：GLORYS 文件索引、区域裁剪和分析/再分析比较工具。
- `plot_fig2_fragility_case.py`：脆弱性案例图。
- `plot_fig4_depth_profiles.py`：真实初始化误差的深度剖面图。
- `plot_fig5_autoregressive_rollout.py`：49 天自回归误差增长图；脚本保留历史文件名，对应当前稿件图 6。

论文图 1 为方法示意图，没有对应的数值计算脚本。扰动幅度图由 `test_online_continue_simsiam.py` 在多个 `--pgd_epsilon` 下得到评估结果后汇总绘制，原工程中没有独立的最终排版脚本。

### 模型、数据和配置

- `model/`：Swin Transformer、ResNet、FNO、ConvLSTM 及统一模型工厂。
- `zmh_train/model_pre_baselines/`：论文跨骨干实验使用的 AFNONet 和旧版 ResNet 实现。
- `data/`：训练/测试数据加载器。
- `utils/metric.py`：MAE、RMSE 等指标。
- `config/`：Swin Transformer、ResNet、旧版 ResNet、FNO 和 AFNONet 配置。

完整逐文件清单见 `MANIFEST.md`。

## 环境

建议使用 Python 3.10 或更高版本，并在适配本机 CUDA 的环境中安装依赖：

```bash
python -m pip install -r requirements.txt
```

PyTorch/CUDA 的具体版本应按本机驱动选择，不建议直接照搬其他机器上的 CUDA wheel。

## 使用前需要提供的外部资源

本归档仅包含代码与配置，不包含：

- GLORYS12V1 训练集和测试集；
- GLORYS 分析场与再分析场原始 NetCDF 数据；
- 预训练及 FCR 模型权重；
- `outputs/` 下的结果缓存和论文成图。

部分原始脚本和 YAML 仍保留实验机器上的绝对路径。运行前请通过命令行参数覆盖数据、权重和输出路径；对于 `train_nw_pacific.py` 和 `test_nw_pacific.py` 中的硬编码数据路径，需要按实际位置修改。

## 常用入口

从本目录运行：

```bash
# 标准模型训练
python train_nw_pacific.py --yaml_config config/en4_Transformer_bg_model3.yaml

# FCR 微调
python train_online_continue_simsiam.py \
  --yaml_config config/en4_Transformer_bg_model3.yaml \
  --ckpt_path /path/to/pretrained_checkpoint.tar \
  --use_pgd \
  --pgd_epsilon 5e-4 \
  --pgd_alpha 1e-4 \
  --pgd_steps 10 \
  --adv_lambda 0.6 \
  --simsiam_lambda 1.0

# 真实初始化误差评估及图 3
python data_process/evaluate_analysis_init_reanalysis_truth_simsiam.py \
  --analysis_root /path/to/glorys_analysis \
  --reanalysis_root /path/to/glorys_reanalysis \
  --before_ckpt /path/to/standard_checkpoint.tar \
  --after_ckpt /path/to/fcr_checkpoint.tar \
  --output_dir /path/to/results \
  --fig3_case_output /path/to/fig3.pdf

# 49 天自回归实验（当前稿件图 6）
python plot_fig5_autoregressive_rollout.py \
  --standard_checkpoint /path/to/standard_checkpoint.tar \
  --fcr_checkpoint /path/to/fcr_checkpoint.tar \
  --glory_path /path/to/test_nw_pacific \
  --output /path/to/figure6.pdf \
  --csv_output /path/to/figure6.csv \
  --npz_output /path/to/figure6.npz
```

其余参数可用相应脚本的 `--help` 查看。

## 归档边界

没有纳入 `extract_sensitive_regions.py` 和 `evaluate_sensitive_region_interventions.py`，因为敏感区域干预实验未出现在当前论文稿件中。也未纳入编译后的论文、图片、数据集、检查点、日志、缓存或 `__pycache__`。
