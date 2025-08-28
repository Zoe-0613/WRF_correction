from __future__ import annotations
from typing import Dict, List


def get_plot_items(
    wrf_t2m_mean, pred_t2m_mean, era_t2m_mean,
    wrf_d2m_mean, pred_d2m_mean, era_d2m_mean,
    wrf_slp_mean, pred_slp_mean, era_slp_mean,
    wrf_u10_mean, pred_u10_mean, era_u10_mean,
    wrf_v10_mean, pred_v10_mean, era_v10_mean,
    bias_wrf_t2, bias_pred_t2,
    bias_wrf_d2, bias_pred_d2,
    bias_wrf_slp, bias_pred_slp,
    bias_wrf_u10, bias_pred_u10,
    bias_wrf_v10, bias_pred_v10
) -> List[Dict]:
    return [
        # T2M
        {"data": wrf_t2m_mean,  "title": "WRF_T2M",       "vmin": -20, "vmax": 5},
        {"data": pred_t2m_mean, "title": "Pred_T2M",      "vmin": -20, "vmax": 5},
        {"data": era_t2m_mean,  "title": "ERA5_T2M",      "vmin": -20, "vmax": 5},
        {"data": bias_wrf_t2,   "title": "Bias_WRF_T2M",  "vmin": -10, "vmax": 10},
        {"data": bias_pred_t2,  "title": "Bias_Pred_T2M", "vmin": -10, "vmax": 10},

        # D2M
        {"data": wrf_d2m_mean,  "title": "WRF_D2M",       "vmin": -30, "vmax": 0},
        {"data": pred_d2m_mean, "title": "Pred_D2M",      "vmin": -30, "vmax": 0},
        {"data": era_d2m_mean,  "title": "ERA5_D2M",      "vmin": -30, "vmax": 0},
        {"data": bias_wrf_d2,   "title": "Bias_WRF_D2M",  "vmin": -20, "vmax": 20},
        {"data": bias_pred_d2,  "title": "Bias_Pred_D2M", "vmin": -20, "vmax": 20},

        # SLP
        {"data": wrf_slp_mean,  "title": "WRF_SLP",       "vmin": 1000, "vmax": 1030},
        {"data": pred_slp_mean, "title": "Pred_SLP",      "vmin": 1000, "vmax": 1030},
        {"data": era_slp_mean,  "title": "ERA5_SLP",      "vmin": 1000, "vmax": 1030},
        {"data": bias_wrf_slp,  "title": "Bias_WRF_SLP",  "vmin": -10,  "vmax": 10},
        {"data": bias_pred_slp, "title": "Bias_Pred_SLP", "vmin": -10,  "vmax": 10},

        # U10
        {"data": wrf_u10_mean,  "title": "WRF_U10",       "vmin": 0, "vmax": 8},
        {"data": pred_u10_mean, "title": "Pred_U10",      "vmin": 0, "vmax": 8},
        {"data": era_u10_mean,  "title": "ERA5_U10",      "vmin": 0, "vmax": 8},
        {"data": bias_wrf_u10,  "title": "Bias_WRF_U10",  "vmin": -5, "vmax": 5},
        {"data": bias_pred_u10, "title": "Bias_Pred_U10", "vmin": -5, "vmax": 5},

        # V10
        {"data": wrf_v10_mean,  "title": "WRF_V10",       "vmin": -5, "vmax": 5},
        {"data": pred_v10_mean, "title": "Pred_V10",      "vmin": -5, "vmax": 5},
        {"data": era_v10_mean,  "title": "ERA5_V10",      "vmin": -5, "vmax": 5},
        {"data": bias_wrf_v10,  "title": "Bias_WRF_V10",  "vmin": -5, "vmax": 5},
        {"data": bias_pred_v10, "title": "Bias_Pred_V10", "vmin": -5, "vmax": 5},
    ]