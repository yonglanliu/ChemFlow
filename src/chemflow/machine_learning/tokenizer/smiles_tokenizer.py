from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "seyonec/ChemBERTa_zinc250k_v2_40k"
)

smiles = "CC(=O)Oc1ccccc1"

tokens = tokenizer.tokenize(smiles)

print(tokens)

ids = tokenizer.encode(smiles)
vocab_size = tokenizer.vocab_size

print(ids)
print(vocab_size)
#print(tokenizer.get_vocab())
print(tokenizer.convert_ids_to_tokens(ids))