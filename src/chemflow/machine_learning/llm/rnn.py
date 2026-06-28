import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SmilesLSTMGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        parameter_init: str = "lstm_default",
    ):
        super().__init__()
        """A simple LSTM-based SMILES generator.
        Args:
            vocab_size (int): Size of the vocabulary.
            pad_token_id (int): Token ID for padding.
            bos_token_id (int): Token ID for beginning of sequence.
            eos_token_id (int): Token ID for end of sequence.
            embedding_dim (int): Dimension of the token embeddings.
            hidden_dim (int): Dimension of the LSTM hidden state.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate for the LSTM layers.
            parameter_init (str): Method for parameter initialization.
        """

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Encoder layer: token id -> vector
        self.token_encoder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )

        # Language model layer
        self.language_model = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Language modeling head
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        self._initialize_parameters(parameter_init)

    def _initialize_parameters(self, method: str = "lstm_default"):
        if method == "none":
            return

        nn.init.normal_(self.token_encoder.weight, mean=0.0, std=0.02)

        for name, param in self.language_model.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

            elif "bias" in name:
                nn.init.zeros_(param)

                # LSTM gate order: input, forget, cell, output
                hidden_size = self.language_model.hidden_size
                param.data[hidden_size:2 * hidden_size].fill_(1.0)

        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.zeros_(self.lm_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        x = self.token_encoder(input_ids)
        output, hidden = self.language_model(x, hidden)
        logits = self.lm_head(output)
        return logits, hidden

    def sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature

        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(values, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            next_token = indices.gather(-1, sampled)
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        return next_token

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        max_length: int = 100,
        condition_ids: Optional[torch.Tensor] = None,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        device: Optional[str] = None,
    ):
        self.eval()

        if device is None:
            device = next(self.parameters()).device

        # Unconditional generation:
        # prompt = <bos>
        if condition_ids is None:
            input_ids = torch.full(
                (batch_size, 1),
                self.bos_token_id,
                dtype=torch.long,
                device=device,
            )

        # Conditional generation:
        # prompt = <bos> <condition_1> <condition_2> ...
        else:
            condition_ids = condition_ids.to(device)

            if condition_ids.dim() == 1:
                condition_ids = condition_ids.unsqueeze(0)

            batch_size = condition_ids.size(0)

            bos = torch.full(
                (batch_size, 1),
                self.bos_token_id,
                dtype=torch.long,
                device=device,
            )

            input_ids = torch.cat([bos, condition_ids], dim=1)

        hidden = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            logits, hidden = self.forward(input_ids[:, -1:], hidden)
            next_logits = logits[:, -1, :]

            next_token = self.sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
            )

            next_token[finished] = self.pad_token_id
            input_ids = torch.cat([input_ids, next_token], dim=1)

            finished |= next_token.squeeze(-1).eq(self.eos_token_id) # Mark sequences that have generated the <eos> token as finished

            if finished.all():
                break

        return input_ids

    def decode(
        self,
        tokenizer,
        input_ids: torch.Tensor,
        condition_token_ids: Optional[set[int]] = None,
    ):
        smiles_list = []

        special_ids = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }

        if condition_token_ids is not None:
            special_ids.update(condition_token_ids)

        for seq in input_ids:
            ids = [
                token_id
                for token_id in seq.tolist()
                if token_id not in special_ids
            ]

            smiles = tokenizer.decode(ids, skip_special_tokens=True)
            smiles_list.append(smiles)

        return smiles_list

    @classmethod
    def build_model(cls, cfg):
        return cls(
            vocab_size=cfg.vocab_size,
            pad_token_id=cfg.pad_token_id,
            bos_token_id=cfg.bos_token_id,
            eos_token_id=cfg.eos_token_id,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            parameter_init=cfg.parameter_init,
        )