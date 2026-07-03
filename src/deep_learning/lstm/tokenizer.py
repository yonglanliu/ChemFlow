from transformers import AutoTokenizer

from pathlib import Path
from typing import Optional, List

from transformers import AutoTokenizer


class SmilesTokenizer:

    def __init__(
        self,
        tokenizer_name: str = "seyonec/SMILES_tokenized_PubChem_shard00_50k",
        tokenizer_path: Optional[str] = None,
        max_length: int = 128,
        condition_tokens: Optional[List[str]] = None,
    ):
        self.max_length = max_length

        # Load from local directory if provided
        if tokenizer_path is not None and Path(tokenizer_path).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Ensure PAD token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
                or self.tokenizer.sep_token
                or "[PAD]"
            )

        # Add condition tokens
        if condition_tokens is not None:
            self.add_condition_tokens(condition_tokens)

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def add_condition_tokens(
        self,
        condition_tokens: List[str],
    ):
        """
        Add condition tokens such as

        <PI3K_ALPHA>
        <HIGH_PIC50>
        <GOOD_ADMET>
        """

        special_tokens = {
            "additional_special_tokens": condition_tokens
        }

        num_added = self.tokenizer.add_special_tokens(
            special_tokens
        )

        print(f"Added {num_added} condition tokens.")

    def encode(
        self,
        smiles: str,
        conditions: Optional[List[str]] = None,
    ):
        if conditions is None:
            text = smiles
        else:
            text = " ".join(conditions) + " " + smiles

        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
    ):
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def save(self, path: str):
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str):
        return cls(tokenizer_path=path)

# if __name__ == "__main__":
#     tokenizer = SmilesTokenizer(
#         tokenizer_name="seyonec/ChemBERTa_zinc250k_v2_40k",
#         max_length=128,
#         condition_tokens=[
#             "<PI3K_ALPHA>",
#             "<PI3K_BETA>",
#             "<PI3K_DELTA>",
#             "<PI3K_GAMMA>",
#             "<HIGH_PIC50>",
#             "<LOW_PIC50>",
#             "<GOOD_ADMET>",
#             "<BAD_ADMET>",
#         ],
#     )

#     tokenizer_no_condition = SmilesTokenizer(
#         tokenizer_name="seyonec/ChemBERTa_zinc250k_v2_40k",
#         max_length=128,)

#     # Save the tokenizer
#     tokenizer.save("smiles_tokenizer")

#     # Load the tokenizer
#     loaded_tokenizer = SmilesTokenizer.load("smiles_tokenizer")

#     # Test encoding and decoding
#     test_smiles = "CCO"
#     test_conditions = ["<PI3K_ALPHA>", "<HIGH_PIC50>"]
#     encoded = loaded_tokenizer.encode(test_smiles, test_conditions)
#     decoded = loaded_tokenizer.decode(encoded["input_ids"][0])

#     print(f"Original SMILES: {test_smiles}")
#     print(f"Decoded SMILES: {decoded}")
#     print(f"vocab size: {loaded_tokenizer.vocab_size}")
#     print(f'vocab size without condition tokens: {tokenizer_no_condition.vocab_size}')