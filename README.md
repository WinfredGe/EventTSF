<div align="center">

<h2><b>(IJCAI'25) EventTSF: Event-aware Non-stationary Time Series Forecasting</b></h2>

<p>
  <img src="https://github.com/user-attachments/assets/829173cd-4c69-4c11-95bd-4df5befb7dbe" width="70">
</p>

<p>
  <img src="https://img.shields.io/github/last-commit/WinfredGe/EventTSF?color=green" />
  <img src="https://img.shields.io/github/stars/WinfredGe/EventTSF?color=yellow" />
  <img src="https://img.shields.io/github/forks/WinfredGe/EventTSF?color=lightblue" />
  <img src="https://img.shields.io/badge/PRs-Welcome-green" />
</p>

</div>

> ✅ **EventTSF** is an **event-aware framework** for non-stationary time series forecasting.  
> 🧭 It aligns event descriptions with time series to improve forecasting under abrupt distribution shifts.

## 🗞️ Updates / News
- 🚩 **2025**: **EventTSF** has been accepted by *IJCAI 2025*.

## 💫 Introduction
EventTSF provides a PyTorch Lightning implementation for event-aware, non-stationary time series forecasting. It supports event-text and time-series paired data with configurable pipelines via Hydra.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b59d361a-3d02-4030-9018-33390a5b8d9c" height="360" />
</p>

## 📑 Datasets
Download preprocessed datasets:
```bash
wget https://drive.google.com/file/d/132GkrVsrEXqCO6MxBJ3Lg2QezO5JFat3/view?usp=sharing
```

Extract to the project directory:
```bash
unzip datasets.zip -d data/download/
```

## 🚀 Get Started

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
2. Download and unzip the datasets into `data/download/`.
3. Run `bash run.sh` or the individual dataset commands above.

## 🙏 Acknowledgements
- [ArXiv:2407.01392](https://arxiv.org/abs/2407.01392) for the autoregressive diffusion inspiration.
- [ArXiv:2411.09502](https://arxiv.org/abs/2411.09502) for the dynamic noise adjustment design.

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
