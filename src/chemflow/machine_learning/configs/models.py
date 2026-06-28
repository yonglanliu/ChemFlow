from dataclasses import dataclass
from types import SimpleNamespace

@dataclass
class LSTMConfig:
    vocab_size: int = 10000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.2

    parameter_init: str = "lstm_default"