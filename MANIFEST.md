# 文件清单

## 核心入口

- `adversarial_generation.py`
- `train_nw_pacific.py`
- `train_online_continue_simsiam.py`
- `test_nw_pacific.py`
- `test_online_continue_simsiam.py`
- `plot_fig2_fragility_case.py`
- `plot_fig4_depth_profiles.py`
- `plot_fig5_autoregressive_rollout.py`

## 配置

- `config/en4_Transformer_bg_model3.yaml`
- `config/en4_ResNet_bg_model.yaml`
- `config/en4_LegacyResNet_bg_model.yaml`
- `config/en4_FNO_bg_model.yaml`
- `config/en4_AFNONet_bg_model.yaml`

## 数据加载与真实初始化误差评估

- `data/dataload_en4_profile_bg_train.py`
- `data/dataload_en4_profile_bg_samp_gen.py`
- `data_process/compare_glory_analysis_reanalysis.py`
- `data_process/evaluate_analysis_init_reanalysis_truth_simsiam.py`

## 模型

- `model/model_factory.py`
- `model/swinunet.py`
- `model/swinunet_v2.py`
- `model/resnet_baseline.py`
- `model/fno_baseline.py`
- `model/convlstm_baseline.py`
- `zmh_train/model_pre_baselines/afnonet/afnonet.py`
- `zmh_train/model_pre_baselines/afnonet/img_utils.py`
- `zmh_train/model_pre_baselines/resnet/resnet.py`
- `zmh_train/model_pre_baselines/resnet/cnn_blocks.py`

## 指标与说明

- `utils/metric.py`
- `README.md`
- `requirements.txt`
- `MANIFEST.md`

共 31 个归档文件，其中 28 个代码/配置文件，3 个说明或依赖文件。
