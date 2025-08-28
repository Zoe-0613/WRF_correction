from __future__ import annotations
import argparse, os
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


VARS = ["t2m", "d2m", "slp", "u10", "v10"]  # 通道顺序固定为 0..4


def _read_triplet(base_path: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 pred/wrf/era 三个文件，形状均为 (T, N, 5, H, W)
    """
    root = os.path.join(base_path, "data_best_model")
    pred = np.load(os.path.join(root, f"pred_{split}.npy"))
    wrf  = np.load(os.path.join(root, f"wrf_{split}.npy"))
    era  = np.load(os.path.join(root, f"era_{split}.npy"))
    return pred, wrf, era


def _hourly_metrics(x: np.ndarray, y: np.ndarray) -> Tuple[List[float], List[float], List[float], float]:
    """
    x/y: (T,N,H,W) —— 某一变量的两个序列
    返回: (rmse_list, bias_list, mae_list, pcc_all)
    """
    T = x.shape[0]
    rmse, bias, mae = [], [], []
    pcc = pearsonr(x.reshape(-1), y.reshape(-1))[0] if x.size > 0 else np.nan
    for h in range(T):
        diff = np.nan_to_num(x[h] - y[h], nan=0.0)
        bias.append(float(np.mean(diff)))
        rmse.append(float(np.sqrt(np.mean(diff ** 2))))
        mae.append(float(np.mean(np.abs(diff))))
    return rmse, bias, mae, pcc


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def plot_hourly(base_path: str, split: str = "val") -> str:
    pred, wrf, era = _read_triplet(base_path, split)
    T, N, C, H, W = pred.shape
    save_dir = os.path.join(base_path, "fig")
    _ensure_dir(save_dir)

    rows = []
    # 逐变量计算
    for c, var in enumerate(VARS):
        # (T,N,H,W)
        xp, xw, xe = pred[:, :, c], wrf[:, :, c], era[:, :, c]

        rmse_p, bias_p, mae_p, pcc_p = _hourly_metrics(xp, xe)
        rmse_w, bias_w, mae_w, pcc_w = _hourly_metrics(xw, xe)

        # 写入每小时
        for h in range(T):
            rows.append({
                "Variable": var.upper(), "Hour": h + 1,
                "WRF_RMSE": rmse_w[h], "WRF_Bias": bias_w[h], "WRF_MAE": mae_w[h],
                "Pred_RMSE": rmse_p[h], "Pred_Bias": bias_p[h], "Pred_MAE": mae_p[h],
            })
        # 汇总
        rows.append({
            "Variable": var.upper(), "Hour": "Avg",
            "WRF_RMSE": float(np.mean(rmse_w)), "WRF_Bias": float(np.mean(bias_w)), "WRF_MAE": float(np.mean(mae_w)),
            "WRF_PCC": pcc_w,
            "Pred_RMSE": float(np.mean(rmse_p)), "Pred_Bias": float(np.mean(bias_p)), "Pred_MAE": float(np.mean(mae_p)),
            "Pred_PCC": pcc_p,
        })

        # 绘图（4联图）
        hours = np.arange(1, T + 1)
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        # (1) 序列均值对比（对 N/H/W 求均值）
        axes[0].plot(hours, xe.mean(axis=(1, 2, 3)), label="ERA5", marker="o")
        axes[0].plot(hours, xw.mean(axis=(1, 2, 3)), label="WRF", marker="o")
        axes[0].plot(hours, xp.mean(axis=(1, 2, 3)), label="Processed", marker="o")
        axes[0].set_title(f"{var.upper()} mean vs hour"); axes[0].set_xlabel("Hour"); axes[0].legend()

        # (2) MAE
        axes[1].plot(hours, mae_w, label="WRF MAE", marker="o")
        axes[1].plot(hours, mae_p, label="Processed MAE", marker="o")
        axes[1].set_title("MAE"); axes[1].set_xlabel("Hour"); axes[1].legend()

        # (3) Bias
        axes[2].plot(hours, bias_w, label="WRF MBE", marker="o")
        axes[2].plot(hours, bias_p, label="Processed MBE", marker="o")
        axes[2].axhline(0, lw=1)
        axes[2].set_title("MBE"); axes[2].set_xlabel("Hour"); axes[2].legend()

        # (4) RMSE
        axes[3].plot(hours, rmse_w, label="WRF RMSE", marker="o")
        axes[3].plot(hours, rmse_p, label="Processed RMSE", marker="o")
        axes[3].set_title("RMSE"); axes[3].set_xlabel("Hour"); axes[3].legend()

        fig.suptitle(f"{var.upper()} — hourly comparison ({split})")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{var}_total_comparison_4.png"), dpi=200)
        plt.close(fig)

    csv_path = os.path.join(save_dir, "results_full.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot hourly metrics and comparisons from (T,N,5,H,W) npy files.")
    p.add_argument("--base_path", required=True, help="./save/wrf_July")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    out = plot_hourly(a.base_path, a.split)
    print(f"Saved metrics to: {out}")
