from chemflow.cli.main import build_parser


def test_lstm_training_cli_route():
    args = build_parser().parse_args(["train", "lstm", "reference.toml"])

    assert args.command == "train"
    assert args.model == "lstm"
    assert args.config == "reference.toml"
    assert args.func.__name__ == "train_lstm"
