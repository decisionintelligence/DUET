

# <img src="figures/duet.png" alt="Image description" style="width:40px;height:30px;"> DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting

[![KDD](https://img.shields.io/badge/KDD'25-DUET-orange)](https://arxiv.org/pdf/2412.10859)  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-blue)](https://pytorch.org/)  ![Stars](https://img.shields.io/github/stars/decisionintelligence/DUET)  

This code is the official PyTorch implementation of our KDD'25 paper: [DUET](https://arxiv.org/pdf/2412.10859): Dual Clustering Enhanced Multivariate Time Series Forecasting.

If you find this project helpful, please don't forget to give it a ⭐ Star to show your support. Thank you!

> [!IMPORTANT]
> DUET has released the results of a long-term forecasting task with unified hyperparameters, where the **input length is fixed at 96** for all experiments. Click [here](https://github.com/decisionintelligence/DUET/blob/main/figures/DUET_unified_seq_len_96.pdf) to view the results, and click [here](https://github.com/decisionintelligence/DUET/blob/main/scripts/multivariate_forecast/DUET_unified_seq_len_96.sh) to view the script for reproducing the results.

🚩 News (2025.12) 🎉Time series model ([DUET](https://arxiv.org/pdf/2412.10859))  was ranked **[1st of the most influential papers](https://resources.paperdigest.org/2025/09/most-influential-kdd-papers-2025-09-version/)** in KDD 2025🎉

🚩 News (2025.04) [Introduction video](https://dl.acm.org/doi/10.1145/3690624.3709325) about DUET (in English).

🚩 News (2024.12) DUET has been included in the time series forecasting benchmark [TFB](https://github.com/decisionintelligence/TFB) and the time series analytics leaderboard [OpenTS-Bench](https://decisionintelligence.github.io/OpenTS/leaderboards/#multivariate_forecasting).

🚩 News (2024.11) DUET has been accepted by SIGKDD 2025.


## Introduction

**DUET**,  which introduces a <ins>**DU**</ins>al clustering on the temporal and channel dimensions to <ins>**E**</ins>nhance multivariate <ins>**T**</ins>ime series forecasting. Specifically, it clusters sub-series into fine-grained distributions with the **TCM** to better model the heterogeneity of temporal patterns. It also utilizes a Channel-Soft-Clustering strategy and captures the relationships among channels with the **CCM**. Euipped with the dual clustering mechanism, DUET rationally harnesses the spectrum of information from both the temporal and channel dimensions, thus forecasting more accruately.  

<div align="center">
<img alt="Logo" src="figures/overview.png" width="100%"/>
</div>

The important components of DUET: (a) Distribution Router; (b) Linear Pattern Extractor; (c) Learnable Distance Metric; (d) Fusion Module.
<div align="center">
<img alt="Logo" src="figures/detailed_structures.png" width="100%"/>
</div>


## Quickstart

> [!IMPORTANT]
> this project is fully tested under python 3.8, it is recommended that you set the Python version to 3.8.
1. Requirements

Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

```shell
pip install -r requirements.txt
```

2. Data preparation

You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link) or [Baidu Drive](https://pan.baidu.com/s/1ycq7ufOD2eFOjDkjr0BfSg?pwd=bpry). Then place the downloaded data under the folder `./dataset`. 

3. Train and evaluate model

- To see the model structure of DUET,  [click here](./ts_benchmark/baselines/duet/models/duet_model.py).
- We provide all the experiment scripts for DUET and other baselines under the folder `./scripts/multivariate_forecast`.  For example you can reproduce all the experiment results as the following script:

```shell
sh ./scripts/multivariate_forecast/ETTh1_script/DUET.sh
```



## Results
We utilize the Time Series Forecasting Benchmark ([TFB](https://github.com/decisionintelligence/TFB)) code repository as a unified evaluation framework, providing access to **all baseline codes, scripts, and results**. Following the settings in TFB, we do not apply the **"Drop Last"** trick to ensure a fair comparison.

### Results of unified hyperparameters with a fixed look-back window length of 96
**Unified hyperparameter results** for the long-term forecasting task. **We fix the look-back window length as 96 for all experiments.**

<div align="center">
<img alt="Logo" src="figures/DUET_unified_seq_len_96.png" width="50%"/>
</div>


### Results of comprehensive parameter searches
Results from **comprehensive parameter searches** for the long-term forecasting task. The look-back window underwent testing with lengths **36 and 104** for FredMd, NASDAQ, NYSE, NN5, ILI, Covid-19, and Wike2000, and **96, 336, and 512** for all other datasets. **We search for the best results from these look-back windows and report the best results.**


Extensive experiments on  25 real-world datasets from 10 different application domains, demonstrate that DUET achieves state-of-the-art~(SOTA) performance. We show the main results of the 10 commonly-used datasets below, click [here](./figures/other_results.png) to see the results for the remaining 15 datasets:

<div align="center">
<img alt="Logo" src="figures/performance.png" width="50%"/>
</div>



<div align="center">
<img alt="Logo" src="figures/duet_full_results.png" width="100%"/>
</div>

## FAQ

1. How to use Pycharm to run code？

When running under pycharm，please escape the double quotes, remove the spaces, and remove the single quotes at the beginning and end.

Such as: **'{"d_ff": 512, "d_model": 256, "horizon": 24}' ---> {\\"d_ff\\":512,\\"d_model\\":256,\\"horizon\\":24}**

```shell
--config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args {\"horizon\":24} --model-name "duet.DUET" --model-hyper-params {\"batch_size\":8,\"dropout\":0.15,\"fc_dropout\":0,\"d_ff\":1024,\"d_model\":128,\"n_heads\":1,\"e_layers\":2,\"lr\":0.0005,\"horizon\":24,\"seq_len\":104,\"factor\":3,\"lradj\":\"type1\",\"loss\":\"MAE\",\"num_experts\":2,\"k\":2,\"patch_len\":48,\"patience\":5,\"num_epochs\":100,\"CI\":1} --gpus 0 --num-workers 1 --timeout 60000 --save-path "ILI/DUET"
```

2. How to evaluate on your own time series?
   

please see [here](https://github.com/decisionintelligence/TFB/blob/master/docs/tutorials/steps_to_evaluate_your_own_time_series.md#TFB-data-format)!

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{qiu2025duet,
 title     = {DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting},
 author    = {Xiangfei Qiu and Xingjian Wu and Yan Lin and Chenjuan Guo and Jilin Hu and Bin Yang},
 booktitle = {SIGKDD},
 pages     = {1185-1196},
 year      = {2025}
}

@article{qiu2024tfb,
 title   = {TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods},
 author  = {Xiangfei Qiu and Jilin Hu and Lekui Zhou and Xingjian Wu and Junyang Du and Buang Zhang and Chenjuan Guo and Aoying Zhou and Christian S. Jensen and Zhenli Sheng and Bin Yang},
 journal = {Proc. {VLDB} Endow.},
 volume  = {17},
 number  = {9},
 pages   = {2363--2377},
 year    = {2024}
}
```

## Acknowledgement

We would like to thank [Jiyanglin Li](https://github.com/erikalien5595) for identifying a bug in our code repository. The issue has been resolved, and all results have been re-tested to ensure accuracy.

## Contact

If you have any questions or suggestions, feel free to contact:

- [Xiangfei Qiu](https://qiu69.github.io/) (xfqiu@stu.ecnu.edu.cn)
- Xingjian Wu (xjwu@stu.ecnu.edu.cn)


Or describe it in Issues.

