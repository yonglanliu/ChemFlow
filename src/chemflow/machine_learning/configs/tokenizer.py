from dataclasses import dataclass
from typing import List


@dataclass
class TokenizerConfig:

    tokenizer_name: str = "seyonec/ChemBERTa_zinc250k_v2_40k"

    max_length: int = 128

    condition_tokens: List[str] = None

    def __post_init__(self):

        if self.condition_tokens is None:
            self.condition_tokens = [
                "<PI3K_ALPHA>",
                "<PI3K_BETA>",
                "<HIGH_PIC50>",
                "<LOW_PIC50>",
                "<SELECTIVE_ALPHA>",
                "<SELECTIVE_BETA>",
            ]