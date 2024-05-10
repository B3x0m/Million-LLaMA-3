from dataclasses import dataclass
from typing import Optional
import torch

@dataclass 
class ModelArgs:
    dim: int = 256
    n_layers: int = 8 
    n_heads: int = 8
    n_kv_heads: Optional[int] = 4
    vocab_size: int = 20000 
    multiple_of: int = 256 
    ffn_dim_multiplier: Optional[float] = None 
    norm_eps: float = 1e-5
    rope_theta: float = 10000 
    max_batch_size: int = 12 
    max_seq_len: int = 512 
    device: str = "cpu"
    dropout_rate: float = 0.1 
