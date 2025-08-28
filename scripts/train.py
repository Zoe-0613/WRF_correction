from __future__ import annotations
import os, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from src.wrfera.config import get_default_config, Config
from src.wrfera.data import build_loaders
from src.wrfera.eval import evaluate
from src.wrfera.model.TS_UNet import TSUNet
from src.wrfera.utils import EarlyStopping, get_logger, save_checkpoint

def cosine_warmup_lambda(total_epochs=100, warmup_epochs=5):
    def _f(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        p = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * p))
    return _f

def main(args):
    cfg: Config = get_default_config()
    # 覆盖默认目录/超参
    cfg.data.wrf_dir   = args.wrf_dir or cfg.data.wrf_dir
    cfg.data.era5_dir  = args.era5_dir or cfg.data.era5_dir
    cfg.data.geo_path  = args.geo_path or cfg.data.geo_path
    cfg.data.stats_path = args.stats_path or cfg.data.stats_path
    cfg.train.save_dir = args.save_dir or cfg.train.save_dir
    cfg.train.epochs   = args.epochs or cfg.train.epochs
    cfg.train.batch_size = args.batch_size or cfg.train.batch_size
    cfg.train.lr       = args.lr or cfg.train.lr
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_loaders(
        cfg.data.wrf_dir, cfg.data.era5_dir, cfg.data.geo_path, cfg.data.stats_path,
        batch_size=cfg.train.batch_size
    )
    stats = np.load(cfg.data.stats_path)
    target_mean, target_std = stats["mean"], stats["std"]

    model = TSUNet(in_ch=11, out_ch=1, base_ch=cfg.model.base_channel).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr)
    criterion = nn.L1Loss()
    scaler = GradScaler()

    logger = get_logger(os.path.join(cfg.train.save_dir, "logs", "training.log"))
    early = EarlyStopping(patience=cfg.train.patience, min_delta=1e-3)

    best_val_rmse = float("inf")
    best_train_loss = float("inf")

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        running = 0.0

        for history, geo, process, forecast, target in train_loader:
            history, target = history.to(device), target.to(device)
            geo = geo.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(history, geo)            # (T,1,H,W)
                target_t = target.squeeze(0)[:, 0]       # (T,H,W)
                loss = criterion(outputs.squeeze(1), target_t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss)

        # 评估
        train_loss, train_mae, train_rmse, train_bias = evaluate(
            model, train_loader, criterion, target_mean, target_std, device
        )
        val_loss, val_mae, val_rmse, val_bias = evaluate(
            model, val_loader, criterion, target_mean, target_std, device
        )

        logger.info(f"Epoch {epoch}/{cfg.train.epochs}")
        logger.info(f"Train   | Loss {train_loss:.5f} MAE {train_mae:.5f} RMSE {train_rmse:.5f} Bias {train_bias:.5f}")
        logger.info(f"Valid   | Loss {val_loss:.5f}   MAE {val_mae:.5f}   RMSE {val_rmse:.5f}   Bias {val_bias:.5f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            save_checkpoint({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_rmse": val_rmse,
            }, cfg.train.save_dir, "best_model")
            logger.info(f"Saved best model @ epoch {epoch}")

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            save_checkpoint({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
            }, cfg.train.save_dir, "best_train")
            logger.info(f"Saved best-train model @ epoch {epoch}")

        early.step(val_loss)
        if early.early_stop:
            logger.info(f"Early stopping @ epoch {epoch}")
            break

    logger.info("Training complete.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wrf_dir", type=str)
    p.add_argument("--era5_dir", type=str)
    p.add_argument("--geo_path", type=str)
    p.add_argument("--stats_path", type=str)
    p.add_argument("--save_dir", type=str)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    args = p.parse_args()
    main(args)
