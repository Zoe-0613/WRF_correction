from __future__ import annotations
import os, math
from glob import glob
from datetime import datetime, timedelta
from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom


class WRFERADataset(Dataset):
    """
    WRF / ERA5 配对数据集。
    - 输入: 24×(WRF前7通道 + 时间特征4) + 1张归一化地形图 (geo)
    - 目标: ERA5(前5通道,已插值到WRF网格) 与 WRF(前5通道) 的差值 diff
    """
    def __init__(
        self,
        wrf_dir: str,
        era5_dir: str,
        geo_path: str,
        split: str,
        *,
        stats_path: str = "./data/mean_std.npz",
        train_split: float = 0.8,
        val_split: float = 0.1,
        compute_stats_if_train: bool = False,
    ) -> None:
        self.wrf_files = sorted(glob(os.path.join(wrf_dir, "*.npy")))
        self.era5_files = sorted(glob(os.path.join(era5_dir, "*.npy")))
        if len(self.wrf_files) == 0:
            raise FileNotFoundError(f"No WRF .npy found under {wrf_dir}")
        if len(self.era5_files) == 0:
            raise FileNotFoundError(f"No ERA5 .npy found under {era5_dir}")
        if len(self.wrf_files) != len(self.era5_files):
            raise ValueError("WRF and ERA5 file counts mismatch.")

        n = len(self.wrf_files)
        n_train = int(n * train_split)
        n_val = int(n * val_split)
        if split == "train":
            self.wrf_files = self.wrf_files[:n_train]
            self.era5_files = self.era5_files[:n_train]
        elif split == "val":
            self.wrf_files = self.wrf_files[n_train:n_train + n_val]
            self.era5_files = self.era5_files[n_train:n_train + n_val]
        else:  # test
            self.wrf_files = self.wrf_files[n_train + n_val:]
            self.era5_files = self.era5_files[n_train + n_val:]

        self.stats_path = stats_path
        self.geo = self._norm(np.load(geo_path))

        if split == "train" and compute_stats_if_train:
            self.mean, self.std = self._compute_and_save_stats()
        else:
            stats = np.load(stats_path)
            self.mean, self.std = stats["mean"], stats["std"]

    def _bilinear_resize(self, data: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        # data shape: (T, C, H, W)
        _, _, H, W = data.shape
        zh = target_hw[0] / H
        zw = target_hw[1] / W
        return zoom(data, (1, 1, zh, zw), order=1)

    def _compute_and_save_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        # 仅基于训练集，按通道对 (era5_interp - wrf) 计算 mean/std
        diffs: List[np.ndarray] = []
        for wf, ef in zip(self.wrf_files, self.era5_files):
            wrf = np.load(wf)[:, :5]
            era5 = np.load(ef)[:, :5]
            era5_interp = self._bilinear_resize(era5, wrf.shape[2:])
            diffs.append(era5_interp - wrf)
        diffs_cat = np.concatenate(diffs, axis=0)  # (N*24, 5, H, W)
        mean = diffs_cat.mean(axis=(0, 2, 3))
        std = diffs_cat.std(axis=(0, 2, 3))
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        np.savez(self.stats_path, mean=mean, std=std)
        return mean, std

    @staticmethod
    def _time_features(filename: str) -> np.ndarray:
        # sin/cos(hour), sin/cos(day_of_year) 共4维，逐小时 T=24
        base = datetime.strptime(os.path.basename(filename).replace(".npy", ""), "%Y_%m_%d_%H")
        feats = []
        for h in range(24):
            t = base + timedelta(hours=h)
            hour = t.hour
            doy = t.timetuple().tm_yday
            feats.append([
                math.sin(2 * math.pi * hour / 24.0),
                math.cos(2 * math.pi * hour / 24.0),
                math.sin(2 * math.pi * doy / 365.0),
                math.cos(2 * math.pi * doy / 365.0),
            ])
        return np.asarray(feats, dtype=np.float32)  # (24, 4)

    @staticmethod
    def _norm(data: np.ndarray) -> Tensor:
        if data.ndim == 3:
            data = data[0]
        mean, std = data.mean(), data.std()
        return torch.tensor((data - mean) / (std + 1e-6), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.wrf_files)

    def __getitem__(self, idx: int):
        wrf = np.load(self.wrf_files[idx])         # (24, 8, H, W)
        era5 = np.load(self.era5_files[idx])[:, :5]  # (24, 5, h, w)
        wrf_5 = wrf[:, :5]                         # (24, 5, H, W)
        time4 = self._time_features(self.wrf_files[idx])  # (24, 4)

        era5_interp = self._bilinear_resize(era5, wrf.shape[2:])  # (24, 5, H, W)
        diff = era5_interp - wrf_5                                 # (24, 5, H, W)

        # 规范化 WRF 前7通道并拼接时间特征
        x = wrf[:, :7]
        x = (x - x.mean(axis=(0, 2, 3), keepdims=True)) / (x.std(axis=(0, 2, 3), keepdims=True) + 1e-6)
        t = np.repeat(time4[:, :, None, None], x.shape[2], axis=2)
        t = np.repeat(t, x.shape[3], axis=3)  # (24, 4, H, W)
        x = np.concatenate([x, t], axis=1)   # (24, 11, H, W)

        features = torch.tensor(x, dtype=torch.float32)
        geo = self.geo
        wrf_t = torch.tensor(wrf_5, dtype=torch.float32)
        era5_t = torch.tensor(era5_interp, dtype=torch.float32)
        target = torch.tensor(diff, dtype=torch.float32)

        return features, geo, wrf_t, era5_t, target


def build_loaders(
    wrf_dir: str,
    era5_dir: str,
    geo_path: str,
    stats_path: str,
    *,
    batch_size: int = 1,
    train_split: float = 0.8,
    val_split: float = 0.1,
):
    train_set = WRFERADataset(wrf_dir, era5_dir, geo_path, "train",
                              stats_path=stats_path, train_split=train_split,
                              val_split=val_split, compute_stats_if_train=False)
    val_set   = WRFERADataset(wrf_dir, era5_dir, geo_path, "val",
                              stats_path=stats_path, train_split=train_split,
                              val_split=val_split, compute_stats_if_train=False)
    test_set  = WRFERADataset(wrf_dir, era5_dir, geo_path, "test",
                              stats_path=stats_path, train_split=train_split,
                              val_split=val_split, compute_stats_if_train=False)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set,   batch_size=batch_size),
        DataLoader(test_set,  batch_size=batch_size),
    )
