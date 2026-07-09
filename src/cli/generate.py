from pathlib import Path

from src.deep_learning.gpt.generator import load_model_from_checkpoint, generate_smiles


def generate_gpt_smiles(args):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()

    adapter_checkpoint_path = (
        Path(args.adapter_checkpoint).expanduser().resolve()
        if args.adapter_checkpoint is not None
        else None
    )

    model, tokenizer, _, _, _ = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        adapter_checkpoint_path=adapter_checkpoint_path,
    )

    with open(args.output, "w") as f:
        for _ in range(args.num_samples):
            generated_smiles = generate_smiles(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            for smiles in generated_smiles.splitlines():
                smiles = smiles.strip()
                if smiles:
                    f.write(smiles + "\n")
                    #print(smiles)


def add_generate_parser(subparsers):
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate SMILES using a trained GPT model",
    )
    generate_parser.add_argument("gpt", help="Use 'gpt' to specify the GPT model for generation")
    generate_parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint")
    generate_parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer")
    generate_parser.add_argument(
        "--adapter_checkpoint",
        type=str,
        default=None,
        help="Path to the adapter checkpoint (if using LoRA)",
    )
    generate_parser.add_argument("--prompt", type=str, default=None, help="Prompt for generation")
    generate_parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    generate_parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    generate_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    generate_parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling parameter")
    generate_parser.add_argument("--output", type=str, default="generated_smiles.txt", help="Output file for generated SMILES")
    generate_parser.set_defaults(func=generate_gpt_smiles)