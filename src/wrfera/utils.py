from __future__ import annotations
import os, logging
import torch

class EarlyStopping:
    def __init__(self, patience: int = 50, min_delta: float = 1e-3) -> None:
        self.patience, self.min_delta = patience, min_delta
        self.best = None
        self.count = 0
        self.early_stop = False

    def step(self, val: float) -> None:
        if self.best is None or val < self.best - self.min_delta:
            self.best, self.count = val, 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True

def get_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        ch = logging.StreamHandler()
        for h in (fh, ch):
            h.setLevel(logging.INFO)
            h.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(h)
    return logger

def save_checkpoint(state: dict, save_dir: str, tag: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, f"{tag}.pth.tar"))
