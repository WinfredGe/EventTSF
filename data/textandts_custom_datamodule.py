import os
import torch
import json
import warnings
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig
warnings.filterwarnings('ignore')


class Dataset_TextandTS(torch.utils.data.Dataset):
    def __init__(self, root_cfg: DictConfig, split="training"):
        self.root_cfg = root_cfg
        assert split in ['training', 'validation', 'test', 'metadata']
        self.set_type = {'training': 0, 'validation': 1, 'test': 2, 'metadata': 0}[split]
        self.split = split
        self.path = './data/download/embedding/' + self.root_cfg.data.custom_dataset_name
        self.__read_data__()
        if split == "metadata":
            self._save_metadata()
        self._load_metadata()

    def _save_metadata(self):
        metadata_path = Path(self.path, "text_and_ts_custom_metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        torch_dataset = torch.from_numpy(self.full_data_x).float()
        if len(torch_dataset.shape) == 2:
            torch_dataset = torch_dataset.permute((1, 0))
        mean_value = torch.mean(torch_dataset, dim=1).numpy()
        std_value = torch.clamp(torch.std(torch_dataset, dim=1), min=1e-7).numpy()
        train_data_mean = mean_value.tolist()
        train_data_std = std_value.tolist()
        target_dimension = self.full_data_x.shape[1]
        total_samples = self._calculate_total_samples(len(self.full_data_x))
        metadata = {
            "target_dimension": target_dimension,
            "train_data_mean": train_data_mean,
            "train_data_std": train_data_std,
            "total_samples": total_samples,
            "total_timesteps": len(self.full_data_x),
            "seq_len": self.root_cfg.data.seq_len,
            "slice_size": self.root_cfg.data.slice_size
        }
        temp_metadata_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        try:
            with open(temp_metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            temp_metadata_path.replace(metadata_path)
        except Exception as e:
            print(f"Failed to write metadata: {e}")
            if temp_metadata_path.exists():
                temp_metadata_path.unlink()

    def _load_metadata(self):
        metadata_path = Path(self.path, "text_and_ts_custom_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            for k, v in metadata.items():
                setattr(self, k, v)

    def _calculate_total_samples(self, data_length):
        available_length = data_length - self.root_cfg.data.seq_len + 1
        return available_length // self.root_cfg.data.slice_size

    def __read_data__(self):
        file_path = os.path.join(self.path, next(f for f in os.listdir(self.path) if f.endswith('.csv')))
        df_raw = pd.read_csv(file_path)
        cols = list(df_raw.columns)
        cols.remove(self.root_cfg.data.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.root_cfg.data.target]]
        df_data = df_raw[[self.root_cfg.data.target]]
        full_data = df_data.values
        text_data = df_raw[['embedding']]
        full_text_data = np.array([eval(embedding) if isinstance(embedding, str) else embedding
                                   for embedding in text_data['embedding'].values])
        self.full_data_x = full_data
        self.full_data_x_text = full_text_data
        total_samples = self._calculate_total_samples(len(full_data))
        num_train_samples = int(total_samples * 0.8)
        num_test_samples = int(total_samples * 0.1)
        num_vali_samples = total_samples - num_train_samples - num_test_samples
        sample_border1s = [0, num_train_samples, num_train_samples + num_vali_samples]
        sample_border2s = [num_train_samples, num_train_samples + num_vali_samples, total_samples]
        sample_start = sample_border1s[self.set_type]
        sample_end = sample_border2s[self.set_type]
        if sample_start < sample_end:
            first_sample_pos = sample_start * self.root_cfg.data.slice_size
            last_sample_pos = (sample_end - 1) * self.root_cfg.data.slice_size + self.root_cfg.data.seq_len
            data_start = first_sample_pos
            data_end = min(last_sample_pos, len(full_data))
            self.data_x = full_data[data_start:data_end]
            self.data_x_text = full_text_data[data_start:data_end]
            self.sample_start_idx = sample_start
            self.sample_end_idx = sample_end
            self.data_start_pos = data_start
            self.data_end_pos = data_end
        else:
            self.data_x = np.array([]).reshape(0, full_data.shape[1])
            self.data_x_text = np.array([]).reshape(0, full_text_data.shape[1])
            self.sample_start_idx = self.sample_end_idx = 0
            self.data_start_pos = self.data_end_pos = 0
        # print(f"📊 Dataset split info for {self.split}:")
        # print(f"   Total samples: {total_samples}")
        # print(f"   Sample range: [{sample_start}:{sample_end}] ({sample_end - sample_start} samples)")
        # print(f"   Data range: [{data_start}:{data_end}] ({data_end - data_start} timesteps)")
        # print(f"   Expected __len__: {self.__len__()}")

    def __getitem__(self, index: int):
        s_begin = index * self.root_cfg.data.slice_size
        s_end = s_begin + self.root_cfg.data.seq_len
        if s_end > len(self.data_x):
            raise IndexError(f"Sample index {index} out of range. "
                             f"Requested data range [{s_begin}:{s_end}] but data length is {len(self.data_x)}")
        seq_x = self.data_x[s_begin:s_end]
        x_text = self.data_x_text[s_begin:s_end]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        x_text = torch.tensor(x_text, dtype=torch.float32)
        return seq_x, x_text

    def __len__(self) -> int:
        available_length = len(self.data_x) - self.root_cfg.data.seq_len + 1
        if available_length <= 0:
            return 0
        return available_length // self.root_cfg.data.slice_size

    def get_sample_info(self):
        return {
            'split': self.split,
            'sample_range': (self.sample_start_idx, self.sample_end_idx),
            'data_range': (self.data_start_pos, self.data_end_pos),
            'dataset_length': len(self),
            'data_shape': self.data_x.shape,
            'text_shape': self.data_x_text.shape
        }


class TextAndTSCustomDataModule(pl.LightningDataModule):
    def __init__(self, root_cfg: DictConfig):
        super().__init__()
        self.root_cfg = root_cfg
        self.batch_size = root_cfg.data.batch_size
        self.path = './data/download/embedding/' + root_cfg.data.custom_dataset_name
        if os.name == 'nt':
            self.num_workers = 0
        else:
            self.num_workers = 1
        self._prepare_data()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _prepare_data(self):
        metadata_path = Path(self.path,"text_and_ts_custom_metadata.json")
        if not metadata_path.exists():
            print("Creating metadata for text and TS custom dataset...")
            _ = Dataset_TextandTS(self.root_cfg, split="metadata")
            print("Text and TS custom dataset metadata created successfully.")

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Dataset_TextandTS(self.root_cfg, split="training")
            self.val_dataset = Dataset_TextandTS(self.root_cfg, split="validation")
        if stage == 'test' or stage is None:
            self.test_dataset = Dataset_TextandTS(self.root_cfg, split="test")
        # if hasattr(self, 'train_dataset') and self.train_dataset:
        #     print("🔍 Train dataset info:", self.train_dataset.get_sample_info())
        # if hasattr(self, 'val_dataset') and self.val_dataset:
        #     print("🔍 Val dataset info:", self.val_dataset.get_sample_info())
        # if hasattr(self, 'test_dataset') and self.test_dataset:
        #     print("🔍 Test dataset info:", self.test_dataset.get_sample_info())

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True if torch.cuda.is_available() else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True if torch.cuda.is_available() else False
        )