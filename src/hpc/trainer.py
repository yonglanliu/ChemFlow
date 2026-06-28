import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from src.chemflow.machine_learning.llm.rnn import SmilesLSTMGenerator
from src.chemflow.machine_learning.train.trainer_llm import (
    get_lstm_config,
    load_pretrained_for_finetune,
    load_or_cache_dataset,
    run_ce_training,
    run_reward_training,
    build_scheduler,
)
from src.chemflow.machine_learning.configs import (
    LLMTrainingConfig,
    TokenizerConfig,
    GenerationConfig,
    LSTMConfig,
)
from src.chemflow.machine_learning.data.dataset import SmilesDataset


def setup_ddp():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def main():
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    try:
        training_config = LLMTrainingConfig()
        tokenizer_config = TokenizerConfig()
        generation_config = GenerationConfig()
        lstm_config = LSTMConfig()

        work_dir = Path(training_config.work_dir)
        cache_dir = work_dir / Path(training_config.cache_dir)
        checkpoint_dir = work_dir / Path(training_config.checkpoint_dir)

        if is_main_process(rank):
            work_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        dist.barrier()

        if is_main_process(rank):
            print(f"Using {world_size} GPUs")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_name)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

        if tokenizer.bos_token_id is None:
            tokenizer.bos_token = tokenizer.cls_token or tokenizer.eos_token

        if tokenizer.eos_token_id is None:
            tokenizer.eos_token = tokenizer.sep_token or tokenizer.bos_token

        condition_tokens = tokenizer_config.condition_tokens

        if condition_tokens is not None:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": condition_tokens}
            )

        cfg = get_lstm_config(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            lstm_config=lstm_config,
        )

        model = SmilesLSTMGenerator.build_model(cfg)

        if training_config.fine_tune:
            if is_main_process(rank):
                print(f"Fine-tuning from {training_config.pretrained_ckpt_path}")

            model = load_pretrained_for_finetune(
                model=model,
                ckpt_path=training_config.pretrained_ckpt_path,
                device=device,
            )

        model = model.to(device)

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

        train_smiles, val_smiles = load_or_cache_dataset(training_config)

        train_dataset = SmilesDataset(
            smiles_list=train_smiles,
            tokenizer=tokenizer,
            max_length=tokenizer_config.max_length,
            condition_list=None,
            ignore_condition_loss=False,
        )

        val_dataset = SmilesDataset(
            smiles_list=val_smiles,
            tokenizer=tokenizer,
            max_length=tokenizer_config.max_length,
            condition_list=None,
            ignore_condition_loss=False,
        )

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=getattr(training_config, "num_workers", 0),
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=getattr(training_config, "num_workers", 0),
            pin_memory=True,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        scheduler = build_scheduler(
            optimizer=optimizer,
            training_config=training_config,
            total_epochs=training_config.epochs_no_reward,
        )

        ce_history, best_ce_path = run_ce_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            training_config=training_config,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )

        dist.barrier()

        if training_config.epochs_with_reward > 0:
            if is_main_process(rank):
                print(f"Loading best CE model for reward training: {best_ce_path}")

            ckpt = torch.load(best_ce_path, map_location=device)

            unwrap_model(model).load_state_dict(ckpt["model"])

            reward_lr = getattr(
                training_config,
                "reward_learning_rate",
                training_config.learning_rate * 0.1,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=reward_lr,
                weight_decay=training_config.weight_decay,
            )

            scheduler = build_scheduler(
                optimizer=optimizer,
                training_config=training_config,
                total_epochs=training_config.epochs_with_reward,
            )

            rl_history, best_rl_path = run_reward_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                training_config=training_config,
                generation_config=generation_config,
                checkpoint_dir=checkpoint_dir,
                device=device,
                start_epoch=len(ce_history["epoch"]) + 1,
            )

            if is_main_process(rank):
                print(f"Best RL model saved at: {best_rl_path}")

        else:
            if is_main_process(rank):
                print(f"Best CE model saved at: {best_ce_path}")

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()