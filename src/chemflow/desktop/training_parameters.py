"""Model-specific training parameter schemas and configuration serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParameterSpec:
    path: str
    default: Any
    help: str
    kind: str = "text"
    options: tuple[str, ...] = ()
    minimum: float | int | None = None
    maximum: float | int | None = None

    @property
    def section(self) -> str:
        return self.path.rsplit(".", 1)[0] if "." in self.path else "General"

    @property
    def key(self) -> str:
        return self.path.rsplit(".", 1)[-1]

    @property
    def label(self) -> str:
        return self.key.replace("_", " ").title()


@dataclass(frozen=True)
class ModelParameters:
    command: str
    description: str
    extension: str
    parameters: tuple[ParameterSpec, ...]


def text(path, default, help): return ParameterSpec(path, default, help)
def integer(path, default, help, minimum=0, maximum=2_147_483_647): return ParameterSpec(path, default, help, "int", minimum=minimum, maximum=maximum)
def number(path, default, help, minimum=-1e12, maximum=1e12): return ParameterSpec(path, default, help, "float", minimum=minimum, maximum=maximum)
def boolean(path, default, help): return ParameterSpec(path, default, help, "bool")
def choice(path, default, options, help): return ParameterSpec(path, default, help, "choice", tuple(options))
def items(path, default, help): return ParameterSpec(path, default, help, "list")


TASKS = ("regression", "classification", "mixed")
SPLITS = ("random", "random_with_repeated_smiles", "scaffold_balanced", "kennard_stone", "kmeans", "predefined")
WEIGHTING = ("uniform", "sqrt_inverse_frequency", "inverse_frequency")


def property_common() -> tuple[ParameterSpec, ...]:
    return (
        text("BaseConfig.workdir", "./training_run", "Directory receiving checkpoints, metrics, plots, logs, and resolved configuration."),
        choice("BaseConfig.task", "regression", TASKS, "Learning problem. Mixed allows regression and binary-classification targets in one model."),
        text("DatasetConfig.dataset_path", "./dataset.csv", "Training CSV or Parquet file available on the machine running the job."),
        text("DatasetConfig.test_dataset_path", "", "Optional independent test dataset. Set test fraction to zero when this is used."),
        text("DatasetConfig.smiles_column", "SMILES", "Column containing molecular SMILES strings."),
        items("DatasetConfig.target_column", ["target"], "One or more target columns. A single item trains a single-task model."),
        items("DatasetConfig.task_types", [], "For mixed learning, one regression/classification value per target; otherwise leave empty."),
        choice("DatasetConfig.split_type", "scaffold_balanced", SPLITS, "How training and validation/test compounds are separated."),
        text("DatasetConfig.split_column", "", "Column containing train/val/test labels when split type is predefined."),
        number("DatasetConfig.val_fraction", 0.1, "Fraction assigned to validation when a split is generated.", 0.0, 0.9),
        number("DatasetConfig.test_fraction", 0.1, "Fraction assigned to internal testing; use zero with an independent test dataset.", 0.0, 0.9),
    )


def transformer_training(
    epochs: int,
    batch: int,
    *,
    encoder_lr: float = 1e-5,
    patience: int = 8,
) -> tuple[ParameterSpec, ...]:
    return (
        integer("TrainingConfig.seed", 42, "Random seed used for splitting, initialization, and sampling."),
        choice("TrainingConfig.device", "auto", ("auto", "cpu", "cuda", "mps"), "Compute backend; auto selects the best available device."),
        integer("TrainingConfig.devices", 1, "Number of processes/devices used by an externally launched distributed run.", 1, 128),
        choice("TrainingConfig.strategy", "auto", ("auto", "ddp"), "Distributed strategy. Use DDP only when launching multiple processes."),
        integer("TrainingConfig.batch_size", batch, "Samples per optimizer step on each device.", 1, 1_000_000),
        integer("TrainingConfig.num_workers", 0, "Data-loading worker processes per training process.", 0, 256),
        integer("TrainingConfig.num_epochs", epochs, "Maximum number of complete passes over the training set.", 1, 100_000),
        boolean("TrainingConfig.progress_bar", True, "Show live batch and epoch progress."),
        boolean("TrainingConfig.tensorboard", True, "Write TensorBoard event logs during training."),
        text("TrainingConfig.tensorboard_dir", "tensorboard", "TensorBoard directory relative to the work directory."),
        boolean("TrainingConfig.resume", False, "Resume model, optimizer, scheduler, epoch, and random state from a full checkpoint."),
        text("TrainingConfig.resume_checkpoint", "", "Optional explicit full-state checkpoint; blank uses the run's last checkpoint."),
        boolean("TrainingConfig.applicability_only", False, "Skip optimization and rebuild applicability/calibration artifacts."),
        number("TrainingConfig.classification_threshold", 0.5, "Probability cutoff used to convert binary probabilities into labels.", 0.0, 1.0),
        choice("TrainingConfig.task_loss_weighting", "sqrt_inverse_frequency", WEIGHTING, "Policy balancing tasks with different label counts."),
        items("TrainingConfig.task_loss_weights", [], "Optional explicit weights in target-column order; blank uses the policy."),
        number("TrainingConfig.encoder_learning_rate", encoder_lr, "Learning rate for the pretrained molecular encoder.", 0.0, 1.0),
        number("TrainingConfig.head_learning_rate", 1e-4, "Learning rate for the newly initialized prediction head.", 0.0, 1.0),
        number("TrainingConfig.weight_decay", 0.01, "AdamW L2-style regularization coefficient.", 0.0, 10.0),
        number("TrainingConfig.gradient_clip", 1.0, "Maximum gradient norm; controls unstable updates.", 0.0, 1e6),
        integer("TrainingConfig.early_stopping_patience", patience, "Validation epochs without improvement before stopping.", 0, 10_000),
    )


GRAPHORMER = ModelParameters(
    "hf-graphormer", "Graph-transformer property prediction.", ".toml",
    property_common() + (
        boolean("DatasetConfig.remove_hs", True, "Remove explicit hydrogen atoms before graph construction."),
        boolean("DatasetConfig.reorder_atoms", False, "Canonicalize atom ordering before Graphormer featurization."),
        text("ModelConfig.model_name", "clefourrier/graphormer-base-pcqm4mv1", "Hugging Face model identifier or local model directory."),
        text("ModelConfig.checkpoint_path", "~/.cache/chemflow/graphormer/graphormer-base-pcqm4mv1.pt", "Optional local pretrained state dictionary, useful on offline compute nodes."),
        boolean("ModelConfig.local_files_only", True, "Forbid model downloads and use only local files/cache."),
        integer("ModelConfig.max_nodes", 512, "Maximum atoms/nodes allowed in one molecular graph.", 1, 100_000),
        integer("ModelConfig.multi_hop_max_dist", 5, "Maximum graph distance encoded by multi-hop edge features.", 1, 1_000),
        integer("ModelConfig.spatial_pos_max", 1024, "Maximum spatial-position embedding index.", 1, 100_000),
        boolean("ModelConfig.freeze_encoder", False, "Keep all pretrained encoder parameters fixed and train only the head."),
    ) + transformer_training(30, 16),
)


CHEMBERTA = ModelParameters(
    "chemberta", "SMILES-transformer property prediction.", ".toml",
    property_common() + (
        text("ModelConfig.model_name", "DeepChem/ChemBERTa-77M-MLM", "Hugging Face identifier or local ChemBERTa directory."),
        choice("ModelConfig.architecture", "roberta", ("roberta", "auto"), "Model/tokenizer loader family."),
        boolean("ModelConfig.local_files_only", False, "Forbid network access and require a local model/cache."),
        integer("ModelConfig.max_length", 512, "Maximum tokenized SMILES sequence length.", 1, 100_000),
        number("ModelConfig.dropout", 0.1, "Dropout probability; nonzero dropout also enables MC-dropout uncertainty.", 0.0, 0.95),
        boolean("ModelConfig.freeze_encoder", False, "Keep the pretrained encoder fixed and train only the property head."),
    ) + transformer_training(20, 32, encoder_lr=2e-5, patience=5),
)


CHEMELEON = ModelParameters(
    "chemeleon", "Pretrained message-passing property prediction.", ".toml",
    property_common() + (
        integer("BaseConfig.seed", 42, "Random seed used for splitting and training."),
        integer("CheMeleonTrainingConfig.batch_size", 64, "Samples per device and optimizer step.", 1, 1_000_000),
        integer("CheMeleonTrainingConfig.num_workers", 0, "Data-loading worker processes per device.", 0, 256),
        integer("CheMeleonTrainingConfig.num_epochs", 30, "Maximum training epochs.", 1, 100_000),
        choice("CheMeleonTrainingConfig.accelerator", "auto", ("auto", "cpu", "gpu", "mps"), "Lightning compute backend."),
        integer("CheMeleonTrainingConfig.devices", 1, "Devices used by Lightning.", 1, 128),
        choice("CheMeleonTrainingConfig.strategy", "auto", ("auto", "ddp"), "Lightning distributed strategy."),
        integer("CheMeleonTrainingConfig.num_nodes", 1, "Slurm nodes participating in distributed training.", 1, 1_000),
        boolean("CheMeleonTrainingConfig.sync_batchnorm", False, "Synchronize batch-normalization statistics across DDP processes."),
        choice("CheMeleonTrainingConfig.precision", "32-true", ("32-true", "16-mixed", "bf16-mixed"), "Numeric precision used during training."),
        boolean("CheMeleonTrainingConfig.deterministic", True, "Request deterministic algorithms where supported."),
        number("CheMeleonTrainingConfig.gradient_clip_value", 1.0, "Maximum gradient norm.", 0.0, 1e6),
        boolean("CheMeleonTrainingConfig.early_stopping", True, "Stop when validation loss ceases improving."),
        integer("CheMeleonTrainingConfig.early_stopping_patience", 6, "Validation epochs without improvement before stopping.", 0, 10_000),
        boolean("CheMeleonTrainingConfig.class_balance", False, "Balance positive and negative examples for classification."),
        boolean("CheMeleonTrainingConfig.evaluate_test", True, "Evaluate the best checkpoint on the test set."),
        boolean("CheMeleonTrainingConfig.plot_training_history", True, "Save training-history figures."),
        boolean("CheMeleonTrainingConfig.inspect_task_metrics", True, "Calculate and save per-task train/validation metrics each epoch."),
        boolean("CheMeleonTrainingConfig.tensorboard", True, "Write TensorBoard event logs."),
        text("CheMeleonTrainingConfig.tensorboard_dir", "tensorboard", "TensorBoard directory relative to workdir."),
        choice("CheMeleonTrainingConfig.task_loss_weighting", "sqrt_inverse_frequency", WEIGHTING, "Policy balancing tasks with different label counts."),
        items("CheMeleonTrainingConfig.task_loss_weights", [], "Optional explicit weights in target-column order."),
        boolean("CheMeleonTrainingConfig.resume", False, "Continue a full Lightning training checkpoint."),
        text("CheMeleonTrainingConfig.resume_checkpoint", "", "Optional explicit last checkpoint used for continuation."),
        boolean("CheMeleonTrainingConfig.verbose", True, "Print initialization and training diagnostics."),
        text("CheMeleonConfig.pretrained_path", "", "Optional official/custom base checkpoint; blank uses verified ChemFlow defaults."),
        text("CheMeleonConfig.transfer_checkpoint", "", "Fine-tuned checkpoint whose encoder initializes sequential transfer learning."),
        boolean("CheMeleonConfig.transfer_encoder_only", True, "Load only encoder weights from the transfer checkpoint."),
        integer("CheMeleonConfig.freeze_epochs", 3, "Initial epochs that train the prediction head while the encoder stays frozen.", 0, 100_000),
        number("CheMeleonConfig.dropout", 0.1, "Prediction-head dropout probability.", 0.0, 0.95),
        integer("CheMeleonConfig.ffn_hidden_dim", 256, "Width of hidden prediction-head layers.", 1, 1_000_000),
        integer("CheMeleonConfig.ffn_num_layers", 1, "Number of prediction-head layers.", 1, 100),
        integer("CheMeleonConfig.warmup_epochs", 2, "Epochs used to increase the learning rate from its initial value.", 0, 100_000),
        number("CheMeleonConfig.init_lr", 1e-5, "Learning rate at the start of warmup.", 0.0, 1.0),
        number("CheMeleonConfig.max_lr", 2e-4, "Peak learning rate.", 0.0, 1.0),
        number("CheMeleonConfig.final_lr", 1e-5, "Learning rate at the end of training.", 0.0, 1.0),
    ),
)


KERMT = ModelParameters(
    "kermt", "Vendored NVIDIA/Merck KERMT graph-transformer.", ".toml",
    (
        text("BaseConfig.workdir", "./kermt_run", "Directory receiving prepared splits, the launch manifest, and official KERMT outputs."),
        choice("BaseConfig.task", "regression", ("regression",), "KERMT property-prediction task; the current adapter supports regression."),
        text("DatasetConfig.dataset_path", "./dataset.csv", "Training CSV or Parquet dataset."),
        text("DatasetConfig.test_dataset_path", "", "Optional independent test dataset; set test fraction to zero."),
        text("DatasetConfig.smiles_column", "SMILES", "Column containing molecular SMILES strings."),
        items("DatasetConfig.target_column", ["target"], "One or more regression endpoints; missing labels enable native multitask training."),
        choice("DatasetConfig.split_type", "scaffold_balanced", SPLITS, "Split used to prepare KERMT's separate train, validation, and test CSV files."),
        text("DatasetConfig.split_column", "", "Existing train/val/test assignment column for predefined splitting."),
        number("DatasetConfig.val_fraction", 0.1, "Validation fraction.", 0.0, 0.9),
        number("DatasetConfig.test_fraction", 0.1, "Internal test fraction; use zero with an independent test dataset.", 0.0, 0.9),
        text("KERMTConfig.checkpoint_path", "", "Optional local checkpoint; blank downloads the pinned official NVIDIA release on first use."),
        text("KERMTConfig.cache_dir", "", "Optional model cache directory; blank uses ~/.cache/chemflow/kermt/NV-KERMT-70M-v2."),
        text("KERMTConfig.model_repo_id", "nvidia/NV-KERMT-70M-v2", "Hugging Face repository used for automatic pretrained-artifact downloads."),
        boolean("KERMTConfig.local_files_only", False, "Require artifacts to exist in the local cache without network access."),
        text("KERMTConfig.python_executable", "", "Python interpreter containing ChemFlow and KERMT dependencies; blank uses the current interpreter."),
        integer("KERMTConfig.ffn_hidden_size", 700, "Prediction-head hidden width.", 1, 1_000_000),
        integer("KERMTConfig.ffn_num_layers", 3, "Prediction-head layer count.", 1, 100),
        number("KERMTConfig.bond_drop_rate", 0.1, "Bond-feature dropout probability.", 0.0, 1.0),
        number("KERMTConfig.dist_coff", 0.15, "KERMT distance-loss coefficient.", 0.0, 1e6),
        number("KERMTConfig.dropout", 0.0, "Network dropout probability.", 0.0, 0.95),
        boolean("KERMTConfig.self_attention", True, "Enable KERMT self-attention during fine-tuning."),
        boolean("KERMTConfig.freeze_encoder", False, "Keep the pretrained KERMT encoder fixed and train only the prediction head/readout."),
        boolean("KERMTConfig.no_features_scaling", True, "Disable a second scaling pass for normalized RDKit features."),
        boolean("KERMTConfig.use_cuikmolmaker_featurization", False, "Use NVIDIA cuik-molmaker acceleration; leave false for the portable RDKit path."),
        text("KERMTConfig.features_generator", "rdkit_2d_normalized", "Use rdkit_2d_normalized on macOS/CPU or rdkit_2d_normalized_cuik_molmaker on NVIDIA HPC."),
        choice("KERMTConfig.rdkit2d_normalization_type", "fast", ("fast", "standard"), "RDKit2D normalization implementation."),
        integer("TrainingConfig.seed", 42, "Random split and KERMT training seed."),
        integer("TrainingConfig.batch_size", 32, "Molecules per optimization batch.", 1, 1_000_000),
        integer("TrainingConfig.num_epochs", 100, "Maximum fine-tuning epochs.", 1, 100_000),
        integer("TrainingConfig.ensemble_size", 1, "Models trained and averaged by KERMT.", 1, 1_000),
        integer("TrainingConfig.num_folds", 1, "KERMT folds.", 1, 1_000),
        choice("TrainingConfig.metric", "mae", ("mae", "rmse", "mse", "r2"), "Validation metric used by KERMT."),
        integer("TrainingConfig.warmup_epochs", 2, "Learning-rate warmup epochs.", 0, 100_000),
        number("TrainingConfig.init_lr", 1e-5, "Initial learning rate.", 0.0, 1.0),
        number("TrainingConfig.max_lr", 1e-4, "Peak learning rate.", 0.0, 1.0),
        number("TrainingConfig.final_lr", 2e-5, "Final learning rate.", 0.0, 1.0),
        number("TrainingConfig.weight_decay", 0.0, "Weight-decay coefficient.", 0.0, 10.0),
        number("TrainingConfig.encoder_lr_multiplier", 1.0, "Advanced encoder learning-rate multiplier; ignored when freeze_encoder is true.", 0.0, 1000.0),
        boolean("TrainingConfig.resume", False, "Continue model, optimizer, scheduler, and epoch from a full KERMT restart checkpoint."),
        text("TrainingConfig.resume_checkpoint", "", "Optional last_checkpoint.pt or KERMT output directory; blank uses this run's checkpoint(s)."),
        items("TrainingConfig.extra_args", [], "Additional official KERMT command-line arguments."),
        boolean("TrainingConfig.dry_run", False, "Prepare data and audit the command without launching KERMT."),
    ),
)


def language_common(
    prefix: str,
    epochs: int,
    *,
    nested_resume: bool = False,
) -> tuple[ParameterSpec, ...]:
    resume_prefix = f"{prefix}.Resume" if nested_resume else prefix
    return (
        text(f"{prefix}.workdir", "./training_run", "Directory receiving checkpoints, history, plots, and cached data."),
        integer(f"{prefix}.batch_size", 32, "Sequences per optimizer step on each device.", 1, 1_000_000),
        number(f"{prefix}.learning_rate", 1e-4, "AdamW learning rate.", 0.0, 1.0),
        number(f"{prefix}.weight_decay", 0.01, "AdamW weight-decay coefficient.", 0.0, 10.0),
        integer(f"{prefix}.num_epochs", epochs, "Maximum training epochs.", 1, 100_000),
        integer(f"{prefix}.seed", 42, "Random seed."),
        boolean(f"{prefix}.early_stopping", True, "Stop when validation loss does not improve."),
        integer(f"{prefix}.early_stopping_patience", 10, "Validation epochs without improvement before stopping.", 0, 10_000),
        choice(f"{prefix}.scheduler", "cosine", ("none", "linear", "cosine", "exponential", "plateau"), "Learning-rate schedule."),
        number(f"{prefix}.gradient_clip_value", 1.0, "Maximum gradient norm.", 0.0, 1e6),
        integer(f"{prefix}.num_workers", 0, "Data-loader worker processes.", 0, 256),
        boolean(f"{prefix}.plot_training_history", True, "Save loss/perplexity training plots."),
        boolean(f"{resume_prefix}.resume", False, "Resume the full latest training state."),
        text(f"{resume_prefix}.resume_checkpoint", "", "Optional explicit full-state checkpoint."),
    )


def language_data() -> tuple[ParameterSpec, ...]:
    return (
        text("DatasetConfig.dataset_path", "./dataset.parquet", "SMILES training dataset."),
        text("DatasetConfig.smiles_column", "SMILES", "Column containing SMILES."),
        text("DatasetConfig.training_X", "train_smiles", "Cached training-input key."),
        text("DatasetConfig.validation_X", "val_smiles", "Cached validation-input key."),
        text("DatasetConfig.training_y", "train_labels", "Cached training-label key."),
        text("DatasetConfig.validation_y", "val_labels", "Cached validation-label key."),
        number("DatasetConfig.val_fraction", 0.1, "Fraction reserved for validation.", 0.0, 0.9),
        integer("DatasetConfig.max_length", 128, "Maximum tokenized sequence length.", 1, 100_000),
        integer("DatasetConfig.preprocess_batch_size", 100000, "Rows processed per preprocessing chunk.", 1, 100_000_000),
        integer("DatasetConfig.seed", 42, "Dataset splitting seed."),
        text("TokenizerConfig.tokenizer_name", "seyonec/ChemBERTa_zinc250k_v2_40k", "Hugging Face tokenizer identifier or local directory."),
        integer("TokenizerConfig.max_length", 128, "Tokenizer truncation length.", 1, 100_000),
        items("TokenizerConfig.condition_tokens", [], "Optional conditioning tokens added to the vocabulary."),
    )


GPT = ModelParameters(
    "gpt", "Transformer SMILES language model.", ".toml",
    (
        integer("GPTConfig.vocab_size", 0, "Vocabulary size; zero allows the trainer/tokenizer to resolve it."),
        integer("GPTConfig.max_len", 128, "Maximum transformer sequence length.", 1, 100_000),
        integer("GPTConfig.d_model", 256, "Token and hidden-state dimension.", 1, 100_000),
        integer("GPTConfig.n_heads", 8, "Self-attention heads; must divide d_model.", 1, 10_000),
        integer("GPTConfig.n_layers", 6, "Transformer blocks.", 1, 1_000),
        integer("GPTConfig.d_ff", 1024, "Feed-forward hidden dimension.", 1, 1_000_000),
        number("GPTConfig.dropout", 0.1, "Transformer dropout probability.", 0.0, 0.95),
        integer("GPTConfig.pad_token_id", 0, "Padding-token vocabulary ID."),
        integer("GPTConfig.bos_token_id", 0, "Beginning-of-sequence token ID."),
        integer("GPTConfig.eos_token_id", 2, "End-of-sequence token ID."),
        boolean("GPTConfig.use_quant_noise", False, "Apply quantization noise during training."),
        number("GPTConfig.quant_noise_p", 0.0, "Probability of quantization noise.", 0.0, 1.0),
        integer("GPTConfig.quant_noise_block_size", 8, "Block size used by quantization noise.", 1, 100_000),
    ) + language_common("GPTTrainingConfig", 50, nested_resume=True) + (
        boolean("GPTTrainingConfig.Finetune.fine_tune", False, "Enable checkpoint fine-tuning instead of pretraining."),
        text("GPTTrainingConfig.Finetune.base_checkpoint", "", "Base GPT checkpoint for fine-tuning."),
        choice("GPTTrainingConfig.Finetune.fine_tune_method", "lora", ("lora",), "Parameter-efficient fine-tuning method."),
        boolean("GPTTrainingConfig.Finetune.fine_tune_lm_head", True, "Fully train the language-model output head."),
        boolean("GPTTrainingConfig.Finetune.fine_tune_layer_norm", True, "Fully train layer-normalization parameters."),
        choice("GPTTrainingConfig.Finetune.Lora.lora_target", "attention", ("attention", "ffn", "all"), "Transformer submodules receiving LoRA adapters."),
        integer("GPTTrainingConfig.Finetune.Lora.lora_r", 8, "LoRA attention rank.", 1, 10_000),
        integer("GPTTrainingConfig.Finetune.Lora.lora_alpha", 16, "LoRA attention scaling factor.", 1, 1_000_000),
        number("GPTTrainingConfig.Finetune.Lora.lora_dropout", 0.05, "LoRA dropout probability.", 0.0, 0.95),
        boolean("GPTTrainingConfig.Finetune.Lora.lora_use_k_proj", False, "Apply LoRA to attention key projection."),
        integer("GPTTrainingConfig.Finetune.Lora.lora_ffn_r", 4, "LoRA feed-forward rank.", 1, 10_000),
        integer("GPTTrainingConfig.Finetune.Lora.lora_ffn_alpha", 16, "LoRA feed-forward scaling factor.", 1, 1_000_000),
        boolean("GPTTrainingConfig.Finetune.Lora.lora_use_fc2", False, "Apply LoRA to the second feed-forward projection."),
    ) + language_data() + (
        integer("GPTGenerationConfig.max_gen_len", 128, "Default maximum generated sequence length.", 1, 100_000),
        number("GPTGenerationConfig.temperature", 1.0, "Default sampling temperature.", 0.001, 100.0),
        integer("GPTGenerationConfig.top_k", 50, "Default top-k sampling cutoff.", 1, 1_000_000),
        number("GPTGenerationConfig.top_p", 0.95, "Default nucleus-sampling probability.", 0.0, 1.0),
    ),
)


LSTM = ModelParameters(
    "lstm", "Recurrent SMILES language model.", ".toml",
    (
        number("GenerationConfig.temperature", 0.8, "Sampling temperature."),
        integer("GenerationConfig.top_k", 50, "Top-k sampling cutoff.", 1, 1_000_000),
        integer("GenerationConfig.max_length", 128, "Tokenizer input length during generation.", 1, 100_000),
        integer("GenerationConfig.num_samples", 100, "Default number of molecules to generate.", 1, 100_000_000),
        integer("GenerationConfig.max_generation_length", 100, "Maximum newly generated tokens.", 1, 100_000),
        integer("LSTMConfig.vocab_size", 10000, "Vocabulary size before loading the tokenizer.", 1, 100_000_000),
        integer("LSTMConfig.pad_token_id", 0, "Padding-token ID."),
        integer("LSTMConfig.bos_token_id", 1, "Beginning-of-sequence token ID."),
        integer("LSTMConfig.eos_token_id", 2, "End-of-sequence token ID."),
        integer("LSTMConfig.embedding_dim", 256, "Token embedding dimension.", 1, 100_000),
        integer("LSTMConfig.hidden_dim", 512, "Recurrent hidden-state dimension.", 1, 1_000_000),
        integer("LSTMConfig.num_layers", 2, "Stacked recurrent layers.", 1, 1_000),
        number("LSTMConfig.dropout", 0.2, "Dropout between recurrent layers.", 0.0, 0.95),
    ) + language_common("LSTMTrainingConfig", 70) + language_data(),
)


ML = ModelParameters(
    "ml", "Descriptor/fingerprint conventional machine learning.", ".json",
    (
        text("workdir", "./ml_training_run", "Directory receiving fitted models, metrics, predictions, and plots."),
        choice("task_type", "regression", ("regression", "classification"), "Supervised learning problem."),
        boolean("resume", False, "Skip model/seed fits already completed successfully."),
        text("data.data_file", "./dataset.csv", "Input molecular dataset."),
        text("data.X_col", "SMILES", "SMILES/structure column."),
        text("data.y_col", "target", "Target column."),
        integer("data.n_classes", 2, "Number of classes for classification.", 2, 10_000),
        items("featurization.features", ["ecfp4"], "Representations concatenated as model input: ecfp4, rdkit2d, avalon, erg, and others."),
        integer("featurization.fp_bits", 2048, "Fingerprint vector length.", 1, 1_000_000),
        number("featurization.descriptor_correlation_threshold", 0.95, "Remove highly correlated descriptor features within each fold.", 0.0, 1.0),
        boolean("featurization.descriptor_model_selection", False, "Run supervised descriptor selection for linear/SVM estimators."),
        text("featurization.descriptor_selection_threshold", "mean", "Feature-importance cutoff accepted by sklearn SelectFromModel."),
        integer("featurization.descriptor_selection_estimators", 128, "Trees used to estimate descriptor importance.", 1, 100_000),
        choice("data_split.split_method", "scaffold", ("random", "scaffold", "predefined", "cluster", "butina"), "Train/validation/test splitting algorithm."),
        text("data_split.split_column", "split", "Existing assignment column for predefined splitting."),
        number("data_split.test_fraction", 0.1, "Fraction assigned to testing.", 0.0, 0.9),
        number("data_split.valid_fraction", 0.1, "Fraction assigned to validation.", 0.0, 0.9),
        integer("data_split.random_seed", 42, "Splitting seed."),
        boolean("data_split.save_split_data", True, "Save featurized arrays and readable split CSV."),
        text("data_split.save_dir", "./split_data", "Directory receiving reusable split artifacts."),
        text("data_split.split_name", "chemflow_split", "Name recorded with split artifacts."),
        choice("Model.estimator", "LGBMRegressor", ("LGBMRegressor", "RandomForestRegressor", "XGBRegressor", "SVR", "LGBMClassifier", "RandomForestClassifier", "XGBClassifier", "SVC"), "Estimator trained by the workflow."),
        text("Model.model_name", "LightGBM", "Readable output-folder and report label."),
        boolean("hyperparameter_tuning", True, "Run randomized hyperparameter search before final fitting."),
        integer("n_iter", 50, "Random-search candidates.", 1, 1_000_000),
        integer("cv", 5, "Inner cross-validation folds.", 2, 1_000),
        choice("cv_strategy", "scaffold-grouped", ("cv", "scaffold-grouped"), "Inner-CV grouping strategy."),
        integer("cv_random_seed", 42, "Hyperparameter-search seed."),
        integer("n_jobs", -1, "Parallel fitting workers; -1 uses all available CPUs.", -1, 10_000),
        items("scoring_metrics", ["r2", "root_mean_squared_error", "mean_absolute_error", "pearson", "spearman", "kendall"], "Metrics calculated for validation and testing."),
        text("refit_metric", "root_mean_squared_error", "Metric selecting the best hyperparameters."),
    ),
)


MODEL_PARAMETER_SCHEMAS = {
    "Conventional ML": ML,
    "Graphormer": GRAPHORMER,
    "CheMeleon": CHEMELEON,
    "ChemBERTa": CHEMBERTA,
    "KERMT": KERMT,
    "GPT": GPT,
    "LSTM": LSTM,
}


def _nested(values: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path, value in values.items():
        if value == "" or value == []:
            continue
        cursor = result
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return result


def _toml_value(value: Any) -> str:
    if isinstance(value, bool): return "true" if value else "false"
    if isinstance(value, str): return json.dumps(value)
    if isinstance(value, list): return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return repr(value)


def _toml_lines(mapping: dict[str, Any], prefix: str = "") -> list[str]:
    lines: list[str] = []
    scalar = [(key, value) for key, value in mapping.items() if not isinstance(value, dict)]
    nested = [(key, value) for key, value in mapping.items() if isinstance(value, dict)]
    if prefix:
        lines.append(f"[{prefix}]")
    lines.extend(f"{key} = {_toml_value(value)}" for key, value in scalar)
    if scalar and nested: lines.append("")
    for index, (key, value) in enumerate(nested):
        lines.extend(_toml_lines(value, f"{prefix}.{key}" if prefix else key))
        if index < len(nested) - 1: lines.append("")
    return lines


def serialize_configuration(model_label: str, values: dict[str, Any]) -> str:
    """Serialize form values to the native format expected by the trainer."""
    nested = _nested(values)
    if model_label == "Conventional ML":
        model = nested.pop("Model")
        nested["models"] = [{"model_name": model["model_name"], "estimator": model["estimator"]}]
        return json.dumps(nested, indent=2) + "\n"
    return "\n".join(_toml_lines(nested)).rstrip() + "\n"


def write_configuration(model_label: str, values: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialize_configuration(model_label, values), encoding="utf-8")
    return destination
