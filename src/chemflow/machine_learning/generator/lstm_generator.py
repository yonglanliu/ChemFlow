import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.chemflow.machine_learning.llm.rnn import SmilesLSTMGenerator


@torch.no_grad()
def generate(
    model,
    tokenizer,
    max_length=128,
    temperature=0.8,
    top_k=20,
    device="cpu",
):
    model.eval()

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    # Some tokenizers (e.g. RoBERTa/ChemBERTa) may not define bos_token_id
    if bos_id is None:
        bos_id = tokenizer.cls_token_id

    if eos_id is None:
        eos_id = tokenizer.sep_token_id

    input_ids = torch.tensor([[bos_id]], device=device)

    hidden = None
    generated_ids = [bos_id]

    for _ in range(max_length):

        logits, hidden = model(input_ids, hidden)

        logits = logits[:, -1, :] / temperature

        values, indices = torch.topk(logits, top_k)

        probs = F.softmax(values, dim=-1)

        sample = torch.multinomial(probs, 1)

        next_token = indices.gather(-1, sample)

        token_id = next_token.item()

        generated_ids.append(token_id)

        if token_id == eos_id:
            break

        input_ids = next_token

    smiles = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return smiles


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "seyonec/ChemBERTa_zinc250k_v2_40k"
    )

    checkpoint = torch.load(
        "checkpoints/smiles_lstm_best.pt",
        map_location=device,
    )
    cfg = checkpoint["config"]
    model = SmilesLSTMGenerator(**cfg)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    for i in range(20):

        smiles = generate(
            model=model,
            tokenizer=tokenizer,
            max_length=128,
            temperature=0.8,
            top_k=20,
            device=device,
        )

        print(f"{i+1:02d}: {smiles}")