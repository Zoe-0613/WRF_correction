from __future__ import annotations
import argparse, os
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.wrfera.data import WRFERADataset
from src.wrfera.model.ST_UNet import STUNet


def _build_loader(split: str,
                  wrf_dir: str, era5_dir: str, geo_path: str, stats_path: str,
                  batch_size: int = 1) -> DataLoader:
    ds = WRFERADataset(wrf_dir, era5_dir, geo_path, split,
                       stats_path=stats_path, compute_stats_if_train=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def run_inference(ckpt_path: str,
                  save_dir: str,
                  split: str,
                  wrf_dir: str, era5_dir: str, geo_path: str, stats_path: str,
                  device: str = "cuda:0") -> Tuple[str, str, str]:
    """
    产出三个一致形状的文件: pred_*.npy, wrf_*.npy, era_*.npy
    形状: (T, N, 5, H, W). pred 的 ch0 = (模型输出 + wrf ch0), 其它通道复用 wrf 便于整体比较。
    """
    os.makedirs(save_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    loader = _build_loader(split, wrf_dir, era5_dir, geo_path, stats_path, batch_size=1)
    model = STUNet(in_ch=11, out_ch=1, base_ch=32).to(dev)

    ckpt = torch.load(ckpt_path, map_location=dev)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    preds: List[np.ndarray] = []
    wrfs:  List[np.ndarray] = []
    eras:  List[np.ndarray] = []

    for history, geo, process, forecast, _ in loader:
        history = history.to(dev)
        geo     = geo.to(dev)
        process = process.to(dev)    # (B=1, T, 5, H, W)
        forecast= forecast.to(dev)   # (B=1, T, 5, H, W)

        # 模型输出: (T,1,H,W)。我们把它加到 wrf 的 ch0 上形成“校正后”的第1变量
        out = model(history, geo)                    # (T,1,H,W)
        out = torch.nan_to_num(out, nan=0.0)
        proc = torch.nan_to_num(process, nan=0.0)
        fore = torch.nan_to_num(forecast, nan=0.0)

        # 组装 pred: 复制 wrf 的全部 5 通道，但把 ch0 替换为 (out + wrf_ch0)
        pred_sample = proc.clone()                   # (1,T,5,H,W)
        pred_sample[:, :, 0, :, :] = out.squeeze(1) + proc[:, :, 0, :, :]

        # 收集为 numpy，去掉 batch 维
        preds.append(pred_sample.squeeze(0).cpu().numpy())   # (T,5,H,W)
        wrfs.append(proc.squeeze(0).cpu().numpy())           # (T,5,H,W)
        eras.append(fore.squeeze(0).cpu().numpy())           # (T,5,H,W)

    # 堆叠成 (N, T, 5, H, W) 再转置为 (T, N, 5, H, W)
    def pack(arrs: List[np.ndarray]) -> np.ndarray:
        arr = np.stack(arrs, axis=0)                # (N,T,5,H,W)
        return arr.transpose(1, 0, 2, 3, 4)         # (T,N,5,H,W)

    pred_all = pack(preds)
    wrf_all  = pack(wrfs)
    era_all  = pack(eras)

    pred_path = os.path.join(save_dir, f"pred_{split}.npy")
    wrf_path  = os.path.join(save_dir, f"wrf_{split}.npy")
    era_path  = os.path.join(save_dir, f"era_{split}.npy")
    np.save(pred_path, pred_all.astype(np.float32))
    np.save(wrf_path,  wrf_all.astype(np.float32))
    np.save(era_path,  era_all.astype(np.float32))

    return pred_path, wrf_path, era_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference and export (T,N,5,H,W) npy files.")
    p.add_argument("--ckpt", required=True, help="checkpoint .pth(.tar)")
    p.add_argument("--save_dir", required=True, help="output directory for npy files")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--wrf_dir", default="./data/npy_wrf")
    p.add_argument("--era5_dir", default="./data/npy_era5")
    p.add_argument("--geo_path", default="./data/topo.npy")
    p.add_argument("--stats_path", default="./data/mean_std.npz")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_inference(a.ckpt, a.save_dir, a.split,
                  a.wrf_dir, a.era5_dir, a.geo_path, a.stats_path, a.device)
