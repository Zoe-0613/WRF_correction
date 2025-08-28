from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class TransformerCfg:
    num_heads: int = 4
    num_layers: int = 4
    embedding_channels: int = 32 * 4
    attention_dropout_rate: float = 0.1
    dropout_rate: float = 0.1

@dataclass
class ModelCfg:
    base_channel: int = 32
    KV_size: int = (32 * 4) * 4
    KV_size_S: int = (32 * 4)
    window_size: int = 8
    attention_channel: List[int] = field(default_factory=lambda: [64, 128, 256, 256])
    attention_spatial: List[int] = field(default_factory=lambda: [65, 129, 257, 257])
    mlp_layers: List[Dict[str, int]] = field(default_factory=lambda: [
        {"in_channels": 65, "out_channels": 64},
        {"in_channels": 129, "out_channels": 128},
        {"in_channels": 257, "out_channels": 256},
        {"in_channels": 257, "out_channels": 256},
    ])
    down_layers: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 256])
    up_layers: List[Dict[str, int]] = field(default_factory=lambda: [
        {"in_channels": 514, "out_channels": 129},
        {"in_channels": 258, "out_channels": 65},
        {"in_channels": 130, "out_channels": 32},
        {"in_channels": 64,  "out_channels": 32},
    ])

@dataclass
class TrainCfg:
    device: str = "cuda:0"
    epochs: int = 100
    batch_size: int = 1
    lr: float = 2.5e-5
    patience: int = 50
    clip_grad_norm: float = 1.0
    save_dir: str = "save/wrf_July"

@dataclass
class DataCfg:
    wrf_dir: str = "./data/npy_wrf"
    era5_dir: str = "./data/npy_era5"
    geo_path: str = "./data/topo.npy"
    stats_path: str = "./data/mean_std.npz"
    train_split: float = 0.8
    val_split: float = 0.1

@dataclass
class Config:
    transformer: TransformerCfg = field(default_factory=TransformerCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    data: DataCfg = field(default_factory=DataCfg)

def get_default_config() -> Config:
    return Config()
