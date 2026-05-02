<div align="center">

<img src="icon.svg" width="520" />

<h3><b>(IJCAI'26) Event-Aware Non-Stationary Time Series Forecasting</b></h3>

<p>
  <img src="https://img.shields.io/github/last-commit/WinfredGe/EventTSF?color=green" />
  <img src="https://img.shields.io/github/stars/WinfredGe/EventTSF?color=yellow" />
  <img src="https://img.shields.io/github/forks/WinfredGe/EventTSF?color=lightblue" />
  <img src="https://img.shields.io/badge/PRs-Welcome-green" />
</p>

</div>

---

✅ **EventTSF** is a first fine-grained **event-aware framework** for non-stationary time series forecasting.  
🧭 It aligns fine-grained event descriptions with time series to improve forecasting under abrupt distribution shifts.

## 🗞️ Updates / News
- 🚩 **2026**: **EventTSF** has been accepted by *IJCAI 2026*.

## 💫 Introduction

**EventTSF is an event-aware autoregressive diffusion framework that forecasts non-stationary time series by jointly modeling historical observations and textual events, capturing event-induced distribution shifts that traditional time-series-only models often miss.**

<p align="center">
  <img src="https://github.com/user-attachments/assets/829173cd-4c69-4c11-95bd-4df5befb7dbe" height="360" />
</p>

- Scenarios and Highlights:

1. **Event-Aware Forecasting**
EventTSF forecasts future temporal dynamics by incorporating known external events, such as holidays, weather changes, public activities, promotions, or news. It helps models anticipate abrupt distribution shifts that cannot be inferred from historical time series alone.

2. **Multimodal Time Series Modeling**
EventTSF aligns discrete textual events with continuous time series segments, enabling fine-grained interaction between event semantics and temporal patterns. This supports real-world forecasting where numerical signals are strongly influenced by external context.

3. **Robust Probabilistic Prediction**
EventTSF uses an autoregressive diffusion framework with event-aware timestep sampling to generate uncertainty-aware forecasts under non-stationary conditions. It improves robustness when events change the magnitude, variance, or shape of future time series.
<p align="center">
  <img src="https://github.com/user-attachments/assets/b59d361a-3d02-4030-9018-33390a5b8d9c" height="360" />
</p>

- Key Components

1. **Event-Synchronized Representation**: A unified multimodal representation that aligns **fine-grained** discrete textual events with continuous time series segments, enabling fine-grained event–series interaction.

2. **AR-Diffusion Forecaster**: An autoregressive diffusion-based forecasting model that **progressively** generates future time series segments conditioned on **different** historical observations and textual events.

3. **Event-Aware Timestep Scheduler**: A learnable event-conditioned timestep reparameterization mechanism that **adapts diffusion denoising difficulty to event-induced non-stationary dynamics**.

4. **M-U-DiT**: A Multimodal U-shaped Diffusion Transformer that bridges event semantics with **multi-resolution** temporal patterns through down-sampling, up-sampling, and skip connections.

5. **Dataset**: A collection of 7 event-aware forecasting datasets spanning synthetic signals, atmospheric physics, traffic, and weather, designed to evaluate multimodal non-stationary time series forecasting.






## 📑 Datasets
Download preprocessed datasets:
```bash
wget -O datasets.zip "https://drive.google.com/uc?export=download&id=132GkrVsrEXqCO6MxBJ3Lg2QezO5JFat3"
```
> [!NOTE]
> If you open the link above directly, please click the download button.

Extract to the project directory:
```bash
unzip datasets.zip -d data/download/
```

## 🚀 Get Started
## Code Overview
The code structure is as follows:
```
EventTSF/
├── algorithms/
│   └── non_stationary_prediction_algorithm.py
│
├── configs/
│   ├── hydra_config.yaml
│   └── data/
│       ├── atmospheric_physics.yaml
│       ├── electricity_accomodation.yaml
│       ├── new_york_taxi.yaml
│       ├── simulation_sine.yaml
│       ├── traffic_FromNewstoForecast.yaml
│       ├── weather_TimeCAP.yaml
│       ├── weather_TimeCAPnew.yaml
│       └── weather_TimeCAPsan.yaml
│
├── data/
│   ├── download/embedding/
│   └── textandts_custom_datamodule.py
│
├── experiments/
│   ├── base_experiment.py
│   └── sequence_prediction_experiment.py
│
├── metrics/
│   ├── CRPS.py
│   ├── ProbMAE.py
│   ├── ProbMSE.py
│   ├── ProbRMSE.py
│   ├── WQL.py
│   ├── __init__.py
│   └── myCRPS.py
│
├── models/
│   ├── flow_matching_transition.py
│   └── timedit.py
│
├── myutils/
│   ├── __init__.py
│   ├── log_plots.py
│   ├── logger.py
│   └── progress_bar.py
│
├── Dockerfile
├── LICENSE
├── README.md
├── icon.svg
├── main.py
└── run.sh
```


### ① Environment Setup

**Option 1: Using Docker (Recommended)**
```bash
docker build -t eventtsf .
docker run --gpus all -it eventtsf
```

**Option 2: Manual Installation**
For this project, we use **Python 3.11** via **Miniconda** and **PyTorch 2.5.0** with **CUDA 12.1.1** and **cuDNN 8**.

```bash
# Create conda environment
conda create -n eventtsf python=3.11
conda activate eventtsf

# Install dependencies
pip install hydra-core
pip install pytorch_lightning==2.5.0
pip install lightning
pip install wandb
pip install torch==2.5.0
```

### ② Run Experiments
```bash
# Run all experiments
bash run.sh

# Individual datasets
python main.py data=atmospheric_physics
python main.py data=new_york_taxi
python main.py data=simulation_sine
python main.py data=traffic_FromNewstoForecast
python main.py data=electricity_accomodation
python main.py data=weather_TimeCAP
python main.py data=weather_TimeCAPnew
python main.py data=weather_TimeCAPsan
```

### ③ Configuration
Experiment configurations are managed via Hydra. See `configs/` for dataset and algorithm settings.

## 📈 Quick Reproduce
1. Set up the environment and install dependencies.
```
docker build -t eventtsf .
docker run --gpus all -it eventtsf
```
2. Download and unzip the datasets into `data/download/`.
```
mkdir -p data/download
wget -O datasets.zip "https://drive.google.com/uc?export=download&id=132GkrVsrEXqCO6MxBJ3Lg2QezO5JFat3"
unzip datasets.zip -d data/download/
```
3. Run `bash run.sh` or the individual dataset commands above.
```
bash run.sh
```


## 📚Further Reading
1, [**T2S: High-resolution Time Series Generation with Text-to-Series Diffusion Models**](https://arxiv.org/abs/2505.02417), *IJCAI* 2025.

```bibtex
@article{ge2025t2s,
  title={T2S: High-resolution Time Series Generation with Text-to-Series Diffusion Models},
  author={Ge, Yunfeng and Li, Jiawei and Zhao, Yiji and Wen, Haomin and Li, Zhao and Qiu, Meikang and Li, Hongyan and Jin, Ming and Pan, Shirui},
  journal={arXiv preprint arXiv:2505.02417},
  year={2025}
}
```

## 🙏 Acknowledgements
Our implementation adapts [Time-Series-Library](https://github.com/thuml/Time-Series-Library), [research-template](https://github.com/buoyancy99/research-template), and [Meta (Scalable Diffusion Models with Transformers)](https://github.com/facebookresearch/DiT) as the code base and have extensively modified it to our purposes. We thank the authors for sharing their implementations and related resources.

## 🙋 Citation
If you find this resource helpful, please consider starring this repo and citing:
```bibtex
@article{ge2025eventtsf,
  title={EventTSF: Event-Aware Non-Stationary Time Series Forecasting},
  author={Ge, Yunfeng and Jin, Ming and Zhao, Yiji and Li, Hongyan and Du, Bo and Xu, Chang and Pan, Shirui},
  journal={arXiv preprint arXiv:2508.13434},
  year={2025}
}
```
