from __future__ import annotations

from pathlib import Path


def train_ml(args):
    from chemflow.machine_learning.train.train_runner import (
        load_training_config,
        train,
    )

    config_path = Path(args.config).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    training_config = load_training_config(config_path)
    train(training_config)


def train_gpt(args):
    from chemflow.deep_learning.gpt.model import GPT
    from chemflow.deep_learning.gpt.trainer import GPTDDPTrainer
    from chemflow.deep_learning.gpt.train_utils import set_seed

    config_path = Path(args.config).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    set_seed(args.seed)

    trainer = GPTDDPTrainer(
        model=GPT,
        config_path=config_path,
    )

    trainer.train()


def train_hf_graphormer(args):
    """Train the Hugging Face Graphormer model."""
    from chemflow.deep_learning.hf_graphormer.trainer import (
        HuggingFaceGraphormerTrainer,
    )

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    trainer = HuggingFaceGraphormerTrainer(config_path)
    trainer.train()


def train_chemeleon(args):
    from chemflow.deep_learning.chemeleon.trainer import CheMeleonTrainer

    config_path = Path(args.config).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    trainer = CheMeleonTrainer(config_path=config_path)
    trainer.train()


def train_chemberta(args):
    """Fine-tune ChemBERTa for molecular-property prediction."""
    from chemflow.deep_learning.chemberta.trainer import ChemBERTaTrainer

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    ChemBERTaTrainer(config_path).train()


def add_train_parser(subparsers):
    train_parser = subparsers.add_parser(
        "train",
        help="Train ChemFlow models",
    )

    model_subparsers = train_parser.add_subparsers(
        dest="model",
        required=True,
    )

    # ========================================================
    # GPT
    # ========================================================

    gpt_parser = model_subparsers.add_parser(
        "gpt",
        help="Train or fine-tune GPT SMILES model",
    )

    gpt_parser.add_argument(
        "config",
        type=str,
    )

    gpt_parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    gpt_parser.set_defaults(
        func=train_gpt
    )

    # ========================================================
    # Graphormer
    # ========================================================

    hf_graphormer_parser = model_subparsers.add_parser(
        "hf-graphormer",
        help="Fine-tune Hugging Face Graphormer for regression or binary classification",
    )
    hf_graphormer_parser.add_argument("config", type=str)
    hf_graphormer_parser.set_defaults(func=train_hf_graphormer)

    chemberta_parser = model_subparsers.add_parser(
        "chemberta",
        help="Fine-tune ChemBERTa for regression, classification, or mixed multitask prediction",
    )
    chemberta_parser.add_argument("config", type=str)
    chemberta_parser.set_defaults(func=train_chemberta)

    # ========================================================
    # CheMeleon
    # ========================================================

    chemeleon_parser = model_subparsers.add_parser(
        "chemeleon",
        help="Fine-tune single-task or multitask CheMeleon models",
    )

    chemeleon_parser.add_argument(
        "config",
        type=str,
    )

    chemeleon_parser.set_defaults(
        func=train_chemeleon
    )

    # ========================================================
    # Classical ML
    # ========================================================

    ml_parser = model_subparsers.add_parser(
        "ml",
        help=(
            "Train machine learning models from "
            "a YAML/JSON/TOML config"
        ),
    )

    ml_parser.add_argument(
        "config",
        type=str,
    )

    ml_parser.set_defaults(
        func=train_ml
    )
