from dataclasses import dataclass


@dataclass
class GenerationConfig:

    temperature: float = 0.8
    top_k: int = 50
    max_length: int = 128
    num_samples: int = 100
    max_generation_length: int = 100