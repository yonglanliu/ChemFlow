import torch
from torch.utils.data import Dataset

class SmilesDataset(Dataset):
    def __init__(
        self,
        smiles_list,
        tokenizer,
        max_length=128,
        condition_list=None,
        ignore_condition_loss=True,
    ):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.condition_list = condition_list
        self.ignore_condition_loss = ignore_condition_loss

        if condition_list is not None:
            assert len(smiles_list) == len(condition_list)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]

        if self.condition_list is None:
            text = smiles
            num_condition_tokens = 0
        else:
            conditions = self.condition_list[idx]

            if isinstance(conditions, str):
                conditions = [conditions]

            condition_text = " ".join(conditions)
            text = condition_text + " " + smiles
            num_condition_tokens = len(conditions)

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        x = input_ids[:-1]
        y = input_ids[1:]
        mask = attention_mask[:-1]

        labels = y.clone()

        # ignore padding loss
        labels[mask == 0] = -100

        # optional: do not train loss on condition tokens
        if self.condition_list is not None and self.ignore_condition_loss:
            labels[:num_condition_tokens] = -100

        return {
            "input_ids": x,
            "labels": labels,
            "attention_mask": mask,
        }

class SmilesDatasetWithLabels(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length=128):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            smiles,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # next-token prediction
        x = input_ids[:-1]
        y = input_ids[1:]
        mask = attention_mask[:-1]

        return {
            "input_ids": x,
            "labels": y,
            "attention_mask": mask,
            "property_label": torch.tensor(label, dtype=torch.float),
        }

# if __name__ == "__main__":
#     from transformers import AutoTokenizer

#     tokenizer = AutoTokenizer.from_pretrained(
#         "seyonec/ChemBERTa_zinc250k_v2_40k"
#     )

#     smiles_list = ["CC(=O)Oc1ccccc1", "C1CCCCC1", "CCO"]
#     dataset = SmilesDataset(smiles_list, tokenizer)

#     for i in range(len(dataset)):
#         sample = dataset[i]
#         print(f"Sample {i}:")
#         print("Input IDs:", sample["input_ids"])
#         print("Labels:", sample["labels"])
#         print("Attention Mask:", sample["attention_mask"])
#         print()