from __future__ import annotations
import argparse
import os
from typing import List, Tuple

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


VARS = ["T2M", "D2M", "SLP", "U10", "V10"]

# 颜色范围（可按需调整）
COLOR_RANGES = {
    "T2M":  {"vmin": 5,   "vmax": 35,  "bmin": -5, "bmax": 5},
    "D2M":  {"vmin": 0,   "vmax": 30,  "bmin": -5, "bmax": 5},
    "SLP":  {"vmin": 980, "vmax": 1020,"bmin": -5, "bmax": 5},
    "U10":  {"vmin": -5,  "vmax": 5,   "bmin": -2, "bmax": 2},
    "V10":  {"vmin": -5,  "vmax": 5,   "bmin": -2, "bmax": 2},
}


def _read_triplet(base_path: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 pred/wrf/era 三个文件，形状均为 (T, N, 5, H, W)
    """
    root = os.path.join(base_path, "data_best_model")
    pred = np.load(os.path.join(root, f"pred_{split}.npy"))
    wrf  = np.load(os.path.join(root, f"wrf_{split}.npy"))
    era  = np.load(os.path.join(root, f"era_{split}.npy"))
    return pred, wrf, era


def _global_means(arr: np.ndarray) -> List[np.ndarray]:
    """
    对 (T,N,5,H,W) 在 T/N 维上求平均 -> [5 * (H,W)]
    """
    means = arr.mean(axis=(0, 1))  # (5,H,W)
    return [means[i] for i in range(means.shape[0])]  # 5 arrays of (H,W)


def _calc_bias(wrf_means: List[np.ndarray], pred_means: List[np.ndarray], era_means: List[np.ndarray]):
    bias_wrf = [w - e for w, e in zip(wrf_means,  era_means)]
    bias_pred= [p - e for p, e in zip(pred_means, era_means)]
    return bias_wrf, bias_pred


def _load_lonlat(nc_path: str) -> Tuple[np.ndarray, np.ndarray]:
    ds = xr.open_dataset(nc_path)
    # 兼容不同命名
    lat = np.asarray(ds.get("lat", ds.get("latitude")).values)
    lon = np.asarray(ds.get("lon", ds.get("longitude")).values)
    return lon, lat


def _plot_variable_row(
    era: np.ndarray, wrf: np.ndarray, pred: np.ndarray,
    bias_wrf: np.ndarray, bias_pred: np.ndarray,
    lon: np.ndarray, lat: np.ndarray,
    var_name: str, vmin: float, vmax: float, bmin: float, bmax: float,
    shapefile: str | None, save_path: str
) -> None:
    """
    画一行五图：ERA5 | WRF | Pred | Bias_WRF | Bias_Pred
    带两条横向 colorbar（主值域 + bias）
    """
    titles = ["ERA5", "WRF", "Pred", "Bias_WRF", "Bias_Pred"]
    data_list = [era, wrf, pred, bias_wrf, bias_pred]
    group = [0, 0, 0, 1, 1]  # 0 使用主值域色标，1 使用 bias 色标

    fig, axs = plt.subplots(
        nrows=1, ncols=5, figsize=(20, 4),
        subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True
    )

    Lon, Lat = np.meshgrid(lon, lat)
    shp = None
    if shapefile and os.path.exists(shapefile):
        shp = gpd.read_file(shapefile).to_crs(epsg=4326)

    ims = []
    for i, ax in enumerate(axs):
        # 若纬度从北到南，输入 data 已在主函数里统一翻转，这里不用再翻
        if group[i] == 0:
            im = ax.pcolormesh(Lon, Lat, data_list[i], cmap="coolwarm",
                               vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        else:
            im = ax.pcolormesh(Lon, Lat, data_list[i], cmap="bwr",
                               vmin=bmin, vmax=bmax, transform=ccrs.PlateCarree())
        if shp is not None:
            shp.plot(ax=ax, color="none", edgecolor="black", linewidth=0.5)
        ax.set_title(titles[i], fontsize=10)
        ax.set_extent([float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
                      crs=ccrs.PlateCarree())
        ax.axis("off")
        ims.append(im)

    # 两条横向 colorbar
    cbar_ax1 = fig.add_axes([0.03, -0.05, 0.54, 0.03])
    cb1 = fig.colorbar(ims[0], cax=cbar_ax1, orientation="horizontal")
    cb1.set_label(var_name, fontsize=10)

    cbar_ax2 = fig.add_axes([0.62, -0.05, 0.36, 0.03])
    cb2 = fig.colorbar(ims[3], cax=cbar_ax2, orientation="horizontal")
    cb2.set_label("Bias", fontsize=10)

    fig.patch.set_alpha(0.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_maps(
    base_path: str, split: str, lonlat_nc: str, shapefile: str | None
) -> List[str]:
    """
    - 从 (T,N,5,H,W) 生成各变量的五图一行对比：ERA5 / WRF / Pred / Bias_WRF / Bias_Pred
    - 经纬度来自含 'lon'/'lat'（或 'longitude'/'latitude'）的 netCDF
    """
    pred, wrf, era = _read_triplet(base_path, split)

    # 全局平均（T/N 维），得到每个变量一张 (H,W)
    pred_means = _global_means(pred)
    wrf_means  = _global_means(wrf)
    era_means  = _global_means(era)
    bias_wrf, bias_pred = _calc_bias(wrf_means, pred_means, era_means)

    # 经纬度
    lon, lat = _load_lonlat(lonlat_nc)
    # 若纬度是从北到南，则翻转所有图
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        pred_means = [arr[::-1, :] for arr in pred_means]
        wrf_means  = [arr[::-1, :] for arr in wrf_means]
        era_means  = [arr[::-1, :] for arr in era_means]
        bias_wrf   = [arr[::-1, :] for arr in bias_wrf]
        bias_pred  = [arr[::-1, :] for arr in bias_pred]

    out_dir = os.path.join(base_path, "fig")
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    for i, var in enumerate(VARS):
        rng = COLOR_RANGES[var]
        save_path = os.path.join(out_dir, f"{var}_comparison.png")
        _plot_variable_row(
            era=era_means[i], wrf=wrf_means[i], pred=pred_means[i],
            bias_wrf=bias_wrf[i], bias_pred=bias_pred[i],
            lon=lon, lat=lat,
            var_name=var,
            vmin=rng["vmin"], vmax=rng["vmax"],
            bmin=rng["bmin"], bmax=rng["bmax"],
            shapefile=shapefile,
            save_path=save_path
        )
        saved.append(save_path)
        print(f"Saved: {save_path}")
    return saved


def parse_args():
    p = argparse.ArgumentParser(description="Draw rows of ERA/WRF/Pred and Bias maps for each variable.")
    p.add_argument("--base_path", required=True, help="e.g., ./save/wrf_July")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--lonlat_nc", required=True, help="netCDF file with lon/lat or longitude/latitude")
    p.add_argument("--shapefile", default="./data/shapefiles/china.shp", help="optional shapefile path")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    outs = plot_maps(a.base_path, a.split, a.lonlat_nc, a.shapefile)
    for p in outs:
        print(p)
