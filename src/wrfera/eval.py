from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    criterion: nn.Module,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    评估流程：
    - model(history, geo) -> outputs (T,1,H,W)
    - outputs + process[:, :, 0:1]
    - mask 掉 forecast NaN
    - 计算 Loss/MAE/RMSE/Bias
    """
    model.eval()
    tot_loss = tot_mae = tot_sq = tot_bias = 0.0
    n = 0

    for history, geo, process, forecast, _ in data_loader:
        history, process, forecast = history.to(device), process.to(device), forecast.to(device)
        geo = geo.to(device)
        outputs = model(history, geo)  # (T,1,H,W)

        outputs = torch.nan_to_num(outputs, nan=0.0).unsqueeze(0)  # (B=1,T,1,H,W)
        outputs = outputs + torch.nan_to_num(process[:, :, 0:1], nan=0.0, posinf=1e6)

        nan_mask = torch.isnan(forecast[:, :, 0:1])
        forecast_clean = torch.nan_to_num(forecast[:, :, 0:1], nan=0.0, posinf=1e6)
        outputs[nan_mask] = 0.0

        loss = criterion(outputs, forecast_clean)
        mae = nn.functional.l1_loss(outputs, forecast_clean, reduction="mean")

        y_pred = outputs.detach().cpu().numpy().ravel()
        y_true = forecast_clean.detach().cpu().numpy().ravel()

        tot_loss += float(loss)
        tot_mae  += float(mae)
        tot_sq   += mean_squared_error(y_pred, y_true)
        tot_bias += float((forecast_clean - outputs).mean())
        n += 1

    return tot_loss / n, tot_mae / n, float(np.sqrt(tot_sq / n)), tot_bias / n
