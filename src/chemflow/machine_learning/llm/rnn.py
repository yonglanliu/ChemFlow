
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmilesLSTMGenerator(nn.Module):
    def __init__(self, vocab_size: int, pad_token_id: int, embedding_dim: int = 256, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.2):
        """
        LSTM-based generator for SMILES strings.

        Args:
            vocab_size (int): Size of the vocabulary.
            pad_token_id (int): ID of the padding token.
            embedding_dim (int, optional): Dimension of the embedding layer. Defaults to 256.
            hidden_dim (int, optional): Dimension of the LSTM hidden states. Defaults to 512.
            num_layers (int, optional): Number of LSTM layers. Defaults to 2.
            dropout (float, optional): Dropout rate for the LSTM. Defaults to 0.2.
        """
        super(SmilesLSTMGenerator, self).__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim, 
            padding_idx=pad_token_id)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,  # Dimension of the input embeddings
            hidden_size=hidden_dim,  # Dimension of the hidden state
            num_layers=num_layers,  # Number of stacked LSTM layers
            batch_first=True,  # Ensures input and output tensors are of shape (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0
            )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, hidden: tuple = None):
        """
        Forward pass of the LSTM generator.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_length).
            hidden (tuple, optional): Tuple containing the hidden and cell states. Defaults to None.

        Returns:
            torch.Tensor: Output logits of shape (batch_size, seq_length, vocab_size).
            tuple: Updated hidden and cell states.
        """
        embedded = self.embedding(input_ids)  # Shape: (batch_size, seq_length, embedding_dim)
        lstm_out, hidden = self.lstm(embedded, hidden)  # lstm_out shape: (batch_size, seq_length, hidden_dim)
        logits = self.fc(lstm_out)  # Shape: (batch_size, seq_length, vocab_size)
        return logits, hidden

if __name__ == "__main__":
    # Example usage
    vocab_size = 100  # Example vocabulary size
    pad_token_id = 0  # Example padding token ID
    model = SmilesLSTMGenerator(vocab_size=vocab_size, pad_token_id=pad_token_id)

    # Dummy input: batch of 2 sequences, each of length 5
    input_ids = torch.tensor([[1, 2, 3, 4, pad_token_id], [5, 6, pad_token_id, pad_token_id, pad_token_id]])
    
    logits, hidden = model(input_ids)
    print("Logits shape:", logits.shape)  # Expected: (batch_size, seq_length, vocab_size)