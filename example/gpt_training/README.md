# GPT molecular generation tutorial

ChemFlow's GPT module trains a decoder-only language model on SMILES strings.
This directory contains examples for training from scratch, resuming training,
LoRA fine-tuning, distributed training, and molecular generation.

Run all commands from the ChemFlow repository root.

## 1. Activate ChemFlow

```bash
conda activate chemflow
cd /path/to/ChemFlow
chemflow --help
```

The workflow requires PyTorch, Transformers, pandas, PyArrow, and Matplotlib.
The tokenizer is downloaded from Hugging Face on first use and saved in the
training work directory. On an offline machine, set `tokenizer_name` to a
previously downloaded local tokenizer directory.

## 2. Prepare an unlabeled SMILES dataset

GPT training is self-supervised, so only a SMILES column is required. CSV,
Parquet, JSON, and `.smi` files are supported.

```csv
SMILES
CCO
CCN
c1ccccc1
CC(=O)Oc1ccccc1C(=O)O
```

For `.smi` input, place one SMILES on each line. If a line contains additional
space-separated fields, ChemFlow uses its first field.

The loader removes missing and empty SMILES, but it does not perform an RDKit
chemical-validity check. ChemFlow randomly assigns `val_fraction` of each
preprocessing batch to validation. The current GPT loader does not read
predefined train and validation split columns.

## 3. Configure training from scratch

Copy `conf.toml` and update at least `workdir` and `dataset_path`. A practical
minimal configuration is:

```toml
[GPTConfig]
vocab_size = 0
max_len = 128
d_model = 256
n_heads = 8
n_layers = 6
d_ff = 1024
dropout = 0.1
pad_token_id = 0
bos_token_id = 0
eos_token_id = 2
use_quant_noise = false
quant_noise_p = 0.0
quant_noise_block_size = 8

[GPTTrainingConfig]
workdir = "./example/gpt_training/zinc_gpt"
batch_size = 32
learning_rate = 1e-4
weight_decay = 0.01
num_epochs = 50
seed = 42
early_stopping = true
early_stopping_patience = 10
scheduler = "cosine"
gradient_clip_value = 1.0
num_workers = 0
plot_training_history = true

[GPTTrainingConfig.Resume]
resume = false
resume_checkpoint = "./example/gpt_training/zinc_gpt/checkpoints/last_model.pt"

[GPTTrainingConfig.Finetune]
fine_tune = false
base_checkpoint = "./path/to/base_model.pt"
fine_tune_method = "lora"
fine_tune_lm_head = true
fine_tune_layer_norm = true

[GPTTrainingConfig.Finetune.Lora]
lora_target = "attention"
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_use_k_proj = false
lora_ffn_r = 4
lora_ffn_alpha = 16
lora_use_fc2 = false

[DatasetConfig]
dataset_path = "./dataset/zinc_smiles.parquet"
smiles_column = "SMILES"
training_X = "train_smiles"
validation_X = "val_smiles"
training_y = "train_labels"
validation_y = "val_labels"
val_fraction = 0.1
max_length = 128
preprocess_batch_size = 100000
seed = 42

[TokenizerConfig]
tokenizer_name = "seyonec/ChemBERTa_zinc250k_v2_40k"
max_length = 128
condition_tokens = []

[GPTGenerationConfig]
max_gen_len = 128
temperature = 1.0
top_k = 50
top_p = 0.95
```

`vocab_size` and special-token IDs are resolved from the tokenizer during
training. Keep `GPTConfig.max_len`, `DatasetConfig.max_length`, and
`TokenizerConfig.max_length` consistent. The current loader uses
`TokenizerConfig.max_length` for tokenization.

The `training_X`, `validation_X`, `training_y`, and `validation_y` fields are
retained for configuration compatibility but are not used by the current data
loader. Supported scheduler values are `none`, `linear`, `cosine`,
`exponential`, and `plateau`.

## 4. Train the model

Update `example/gpt_training/conf.toml`, then run:

```bash
chemflow train gpt example/gpt_training/conf.toml
```

The trainer automatically selects CUDA, Apple MPS, or CPU. CUDA is recommended
for substantial datasets. Set the reproducibility seed with
`GPTTrainingConfig.seed` in the TOML file.

### Multi-GPU DDP training

For multiple CUDA GPUs on one node:

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  -m chemflow.cli.main \
  train gpt example/gpt_training/conf.toml
```

Set `--nproc_per_node` to the number of GPUs. `train.sh` contains a Slurm
example. Before submitting it, replace the environment activation command and
change its `CONFIG` variable to `example/gpt_training/conf.toml` or another
valid configuration path.

## 5. Training outputs and cache

For `workdir = "./example/gpt_training/zinc_gpt"`, training creates:

```text
example/gpt_training/zinc_gpt/
├── config.json
├── cache/
│   ├── tokenized_manifest.pt
│   ├── tokenizer/
│   ├── train/
│   └── val/
├── checkpoints/
│   ├── best_model.pt
│   ├── last_model.pt
│   ├── history.csv
│   ├── history.json
│   └── history.pt
└── plots/
    ├── learning_rate.png
    ├── loss_curve.png
    └── perplexity_curve.png
```

`best_model.pt` has the lowest validation loss. `last_model.pt` contains the
latest full training state and is the appropriate checkpoint for resuming.

The tokenized cache is reused whenever `tokenized_manifest.pt` exists. Use a
new work directory or remove the old cache before retraining with a different
dataset, tokenizer, maximum length, validation fraction, or condition-token
list.

## 6. Resume interrupted training

Set the total desired epoch count and enable resume:

```toml
[GPTTrainingConfig]
num_epochs = 100

[GPTTrainingConfig.Resume]
resume = true
resume_checkpoint = "./example/gpt_training/zinc_gpt/checkpoints/last_model.pt"
```

Then rerun training:

```bash
chemflow train gpt example/gpt_training/conf.toml
```

The trainer restores the model, optimizer, scheduler, best validation score,
early-stopping state, and history. Setting `resume_checkpoint` alone is not
enough; `resume` must also be `true`.

## 7. LoRA fine-tuning

Use `conf_fine_tune.toml` as the starting point. The base checkpoint must be a
full GPT checkpoint created by the non-fine-tuning workflow.

```toml
[GPTTrainingConfig]
workdir = "./example/gpt_training/my_lora_run"
learning_rate = 1e-4
num_epochs = 20

[GPTTrainingConfig.Resume]
resume = false
resume_checkpoint = "./example/gpt_training/my_lora_run/checkpoints/last_model.pt"

[GPTTrainingConfig.Finetune]
fine_tune = true
base_checkpoint = "./example/gpt_training/zinc_gpt/checkpoints/best_model.pt"
fine_tune_method = "lora"
fine_tune_lm_head = true
fine_tune_layer_norm = true

[GPTTrainingConfig.Finetune.Lora]
lora_target = "attention"
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_use_k_proj = false
```

Update `[DatasetConfig]` for the fine-tuning data. Keep the GPT architecture,
tokenizer, vocabulary, and condition-token list compatible with the base
checkpoint. Start fine-tuning with:

```bash
chemflow train gpt example/gpt_training/conf_fine_tune.toml
```

LoRA training freezes the base model and can additionally train the language
model head and layer-normalization parameters. It writes adapter-only files:

```text
<fine-tune-workdir>/checkpoints/best_adapter.pt
<fine-tune-workdir>/checkpoints/last_adapter.pt
```

The current generation loader reconstructs attention LoRA, so keep
`lora_target = "attention"` for adapters used with the CLI. Generation requires
both the unchanged full base checkpoint and the adapter checkpoint.

## 8. Unconditional generation

Use the checkpoint and tokenizer saved by the same training run:

```bash
mkdir -p example/gpt_training/generated

chemflow generate gpt \
  --checkpoint ./example/gpt_training/zinc_gpt/checkpoints/best_model.pt \
  --tokenizer ./example/gpt_training/zinc_gpt/cache/tokenizer \
  --num_samples 100 \
  --max_new_tokens 100 \
  --temperature 0.8 \
  --top_k 20 \
  --output ./example/gpt_training/generated/smiles.txt
```

The output contains one generated string per line. ChemFlow does not
automatically validate, canonicalize, or deduplicate generated SMILES.

Sampling controls:

- lower `temperature` makes generation more conservative;
- higher `temperature` increases diversity but can reduce validity;
- lower `top_k` restricts sampling to fewer likely next tokens;
- `temperature` must be greater than zero;
- set `top_k` to `0` to disable top-k filtering.

Although `top_p` appears in the TOML, the current generation CLI uses
`temperature` and `top_k`; it does not currently apply `top_p`.

## 9. Generate with a LoRA adapter

Pass the original full base checkpoint plus the fine-tuned adapter:

```bash
chemflow generate gpt \
  --checkpoint ./example/gpt_training/zinc_gpt/checkpoints/best_model.pt \
  --tokenizer ./example/gpt_training/zinc_gpt/cache/tokenizer \
  --adapter_checkpoint ./example/gpt_training/my_lora_run/checkpoints/best_adapter.pt \
  --num_samples 100 \
  --max_new_tokens 100 \
  --temperature 0.8 \
  --top_k 20 \
  --output ./example/gpt_training/generated/lora_smiles.txt
```

Do not pass `best_adapter.pt` as `--checkpoint`; it does not contain the full
base model.

## 10. Conditional generation

Condition tokens can represent a target, property range, assay, or another
category. Add them before training:

```toml
[TokenizerConfig]
tokenizer_name = "seyonec/ChemBERTa_zinc250k_v2_40k"
max_length = 128
condition_tokens = ["<ACTIVE>", "<INACTIVE>"]
```

Prefix each training SMILES with its token:

```csv
SMILES
<ACTIVE>CCO
<INACTIVE>c1ccccc1
```

Generate from a condition with `--prompt`:

```bash
chemflow generate gpt \
  --checkpoint ./example/gpt_training/conditional_gpt/checkpoints/best_model.pt \
  --tokenizer ./example/gpt_training/conditional_gpt/cache/tokenizer \
  --prompt "<ACTIVE>" \
  --num_samples 100 \
  --output ./example/gpt_training/generated/active_smiles.txt
```

Always use the tokenizer saved by that model. A different tokenizer or
condition-token list can change vocabulary indices and make the checkpoint
incompatible.

## 11. Common problems

### Tokenizer download fails

The first run needs access to the configured Hugging Face tokenizer unless it
is already cached. Download it on a connected machine or set `tokenizer_name`
to a local tokenizer directory.

### Training uses stale data

The tokenized cache is loaded without rebuilding when its manifest exists. Use
a new `workdir` or clear that run's `cache` directory after changing the
dataset or tokenization settings.

### CUDA runs out of memory

Reduce `batch_size`, `max_length`, `d_model`, `n_layers`, or `d_ff`. Gradient
accumulation is not currently exposed by the GPT trainer.

### Generation reports a vocabulary or state-dictionary mismatch

Use the tokenizer saved under the same base model's `cache/tokenizer`
directory. For LoRA generation, ensure that the adapter was trained from that
exact base architecture and tokenizer.
