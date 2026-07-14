from __future__ import annotations

from pathlib import Path

from src.deep_learning.gpt.model import GPT
from src.deep_learning.gpt.trainer import GPTDDPTrainer
from src.deep_learning.gpt.train_utils import set_seed


def train_gpt(args):
    config_path = Path(args.config).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    set_seed(args.seed)

    trainer = GPTDDPTrainer(
        model=GPT,
        config_path=config_path,
    )
    trainer.train()


def train_graphormer(args):
    from src.deep_learning.graphormer.trainer import GraphormerDDPTrainer

    config_path = Path(args.config).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    set_seed(args.seed)

    trainer = GraphormerDDPTrainer(
        config_path=config_path,
    )
    trainer.train()


def add_train_parser(subparsers):
    train_parser = subparsers.add_parser(
        "train",
        help="Train ChemFlow models",
    )

    model_subparsers = train_parser.add_subparsers(
        dest="model",
        required=True,
    )

    gpt_parser = model_subparsers.add_parser(
        "gpt",
        help="Train or fine-tune GPT SMILES model",
    )
    gpt_parser.add_argument("config", type=str)
    gpt_parser.add_argument("--seed", type=int, default=42)
    gpt_parser.set_defaults(func=train_gpt)



    graphormer_parser = model_subparsers.add_parser(
        "graphormer",
        help="Train Graphormer model",
    )
    graphormer_parser.add_argument("config", type=str)
    graphormer_parser.add_argument("--seed", type=int, default=42)
    graphormer_parser.set_defaults(func=train_graphormer)