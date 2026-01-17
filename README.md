# (IJCAI'26) EventTSF: Event-aware Non-stationary Time Series Forecasting



## 1.🚀 Quick Start

### 1.1Environment Setup

**Option 1: Using Docker (Recommended)**
```bash
docker build -t eventtsf .
docker run --gpus all -it eventtsf
```

**Option 2: Manual Installation**

For this project, we're using **Python 3.11** via **Miniconda** and **PyTorch 2.5.0** with **CUDA 12.1.1** and **cuDNN 8**.

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

### Dataset Preparation

1. Download preprocessed datasets:
   ```bash
   wget https://drive.google.com/file/d/132GkrVsrEXqCO6MxBJ3Lg2QezO5JFat3/view?usp=sharing
   ```

2. Extract to project directory:
   ```bash
   unzip datasets.zip -d data/download/
   ```


## 2.📊 Experiments

### Quick Reproduction
```bash
# Run all experiments
bash run.sh

# Individual datasets
python main.py data=atmospheric_physics
python main.py data=new_york_taxi
python main.py data=simulation_sine
python main.py data=traffic_FromNewstoForecast
python main.py data=weather_TimeCAP
python main.py data=weather_TimeCAPnew
python main.py data=weather_TimeCAPsan
```

### Configuration
Experiment configurations are managed via Hydra. See `configs/` for specific settings.




