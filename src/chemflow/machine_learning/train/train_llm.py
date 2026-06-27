import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.config import PROJECT_ROOT
from src.chemflow.machine_learning.llm.rnn import SmilesLSTMGenerator
from src.chemflow.machine_learning.data.dataset import SmilesDataset


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits, _ = model(input_ids)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"  step {step}/{len(loader)} | loss={loss.item():.4f}", flush=True)

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits, _ = model(input_ids)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

        total_loss += loss.item()

    return total_loss / len(loader)


if __name__ == "__main__":
    print("1. Script started", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"2. Device: {device}", flush=True)

    tokenizer_name = "seyonec/ChemBERTa_zinc250k_v2_40k"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("3. Tokenizer loaded", flush=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
        print(f"Set pad token to: {tokenizer.pad_token}", flush=True)

    data_path = Path(PROJECT_ROOT) / "data/combined/combined_pivot.parquet"
    print(f"4. Loading data from: {data_path}", flush=True)

    data = pd.read_parquet(data_path)

    smiles_list = (
        data["SMILES"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    smiles_list = [s for s in smiles_list if s]

    print(f"5. Number of SMILES: {len(smiles_list)}", flush=True)
    print(f"6. First SMILES: {smiles_list[0]}", flush=True)

    train_smiles, val_smiles = train_test_split(
        smiles_list,
        test_size=0.1,
        random_state=42,
    )

    print(f"7. Train size: {len(train_smiles)}", flush=True)
    print(f"8. Val size: {len(val_smiles)}", flush=True)

    train_dataset = SmilesDataset(
        smiles_list=train_smiles,
        tokenizer=tokenizer,
        max_length=128,
    )

    val_dataset = SmilesDataset(
        smiles_list=val_smiles,
        tokenizer=tokenizer,
        max_length=128,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )

    print("9. DataLoaders created", flush=True)

    first_batch = next(iter(train_loader))
    print("10. First batch loaded", flush=True)
    print(first_batch["input_ids"].shape, flush=True)
    print(first_batch["labels"].shape, flush=True)

    model = SmilesLSTMGenerator(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    print("11. Model created", flush=True)

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    checkpoint_dir = Path(PROJECT_ROOT) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = 20
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch + 1}/{num_epochs}", flush=True)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_path = checkpoint_dir / "smiles_lstm_best.pt"

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "tokenizer_name": tokenizer_name,
                    "config": {
                        "vocab_size": len(tokenizer),
                        "pad_token_id": tokenizer.pad_token_id,
                        "embedding_dim": 256,
                        "hidden_dim": 512,
                        "num_layers": 2,
                        "dropout": 0.2,
                    },
                    "best_val_loss": best_val_loss,
                },
                save_path,
            )

            print(f"Saved best model to: {save_path}", flush=True)

    print("Training finished.", flush=True)