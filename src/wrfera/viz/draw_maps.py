from __future__ import annotations
import argparse, os
from typing import Dict, List, Tuple
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from .plot_config import get_plot_items


def _read_triplet(base_path: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = os.path.join(base_path, "data_best_model")
    pred = np.load(os.path.join(root, f"pred_{split}.npy"))  # (T,N,5,H,W)
    wrf  = np.load(os.path.join(root, f"wrf_{split}.npy"))
    era  = np.load(os.path.join(root, f"era_{split}.npy"))
    return pred, wrf, era


def _global_means(arr: np.ndarray) -> List[np.ndarray]:
    """
    对 (T,N,5,H,W) 在 T/N 维上求平均 -> [5 * (H,W)]
    """
    means = arr.mean(axis=(0, 1))    # (5,H,W)
    return [means[i] for i in range(means.shape[0])]  # length 5 list


def _draw_map(data: np.ndarray, lon: np.ndarray, lat: np.ndarray,
              vmin: float, vmax: float, title: str,
              shapefile: str | None, out_path: str, dpi: int = 300) -> None:
    fig = plt.figure(figsize=(8, 4.5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # meshgrid
    LON, LAT = np.meshgrid(lon, lat)

    # 底图
    im = ax.pcolormesh(LON, LAT, data, cmap="coolwarm",
                       vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    cb = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.85)
    ax.set_title(title)

    if shapefile and os.path.exists(shapefile):
        shp = gpd.read_file(shapefile).to_crs(epsg=4326)
        shp.plot(ax=ax, color="none", edgecolor="black", linewidth=0.5)

    # 视域
    ax.set_extent([float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
                  crs=ccrs.PlateCarree())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def plot_maps(base_path: str, split: str,
              lonlat_nc: str,
              shapefile: str | None = None) -> List[str]:
    """
    - 从 (T,N,5,H,W) 生成 ERA/WRF/Pred 的全球分布与 Bias 地图（对 T/N 求平均）
    - lon/lat 来自一个包含 'lon','lat' 的 nc 文件。
    """
    pred, wrf, era = _read_triplet(base_path, split)

    # 取全局时空平均后的 5 个变量
    pred_means = _global_means(pred)
    wrf_means  = _global_means(wrf)
    era_means  = _global_means(era)

    # 偏差
    bias_wrf  = [w - e for w, e in zip(wrf_means,  era_means)]
    bias_pred = [p - e for p, e in zip(pred_means, era_means)]

    # 经纬度
    ds  = xr.open_dataset(lonlat_nc)
    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    # 如果数据纬度是从北到南，flip一次与地图配合
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        pred_means = [arr[::-1, :] for arr in pred_means]
        wrf_means  = [arr[::-1, :] for arr in wrf_means]
        era_means  = [arr[::-1, :] for arr in era_means]
        bias_wrf   = [arr[::-1, :] for arr in bias_wrf]
        bias_pred  = [arr[::-1, :] for arr in bias_pred]

    items = get_plot_items(
        wrf_means[0], pred_means[0], era_means[0],
        wrf_means[1], pred_means[1], era_means[1],
        wrf_means[2], pred_means[2], era_means[2],
        wrf_means[3], pred_means[3], era_means[3],
        wrf_means[4], pred_means[4], era_means[4],
        bias_wrf[0],  bias_pred[0],
        bias_wrf[1],  bias_pred[1],
        bias_wrf[2],  bias_pred[2],
        bias_wrf[3],  bias_pred[3],
        bias_wrf[4],  bias_pred[4],
    )

    out_dir = os.path.join(base_path, "fig")
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    for it in items:
        out_path = os.path.join(out_dir, f"{it['title']}.png")
        _draw_map(it["data"], lon, lat, it["vmin"], it["vmax"], it["title"], shapefile, out_path)
        saved.append(out_path)
    return saved


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Draw global maps for WRF/ERA/Pred means and biases.")
    p.add_argument("--base_path", default="./save/wrf_July")
    p.add_argument("--split", default="val", choices=["train","val","test"])
    p.add_argument("--lonlat_nc", default="./data/wrf_eg.nc")
    p.add_argument("--shapefile", default="./data/shapefiles/china.shp")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    outs = plot_maps(a.base_path, a.split, a.lonlat_nc, a.shapefile)
    for p in outs: print("Saved:", p)
