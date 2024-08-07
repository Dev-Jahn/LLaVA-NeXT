import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, HfArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.internvid import InternVidDataset
from .llava_trainer import LLaVATrainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_path: str = field()
    version: Optional[str] = field(default="v0")
    # freeze_backbone: bool = field(default=False)
    # tune_mm_mlp_adapter: bool = field(default=False)
    # vision_tower: Optional[str] = field(default=None)
    # mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    # pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    # mm_projector_type: Optional[str] = field(default='linear')
    # mm_use_im_start_end: bool = field(default=False)
    # mm_use_im_patch_token: bool = field(default=True)
    # mm_patch_merge_type: Optional[str] = field(default='flat')
    # mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    dataset: str = field()
    data_dir: Optional[str] = field(default=None, metadata={"help": "Path to the data directory."})
    lazy_preprocess: bool = False
    fps: int = None
    num_frames: int = 32
    frame_size: int = 336


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    model_max_length: int = field(default=512, metadata={"help": "Maximum seq length."})

    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Prepare dataset
    dataset = get_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.per_device_train_batch_size, collate_fn=collate_fn)

    # Prepare model and tokenizer
    tokenizer, model, image_processor, context_len = utils.load_pretrained_model(
        path, path, name, llava_model="initial",
        attn_implementation=attn_implementation
    )
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Prepare trainer
    trainer = LLaVATrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model(args.output_dir)


def get_dataset(args):
    if args.dataset == "internvid":
        return InternVidDataset(
            root_dir=args.data_dir,
            fps=args.fps,
            max_frames=args.num_frames,
            frame_size=args.frame_size
        )


def collate_fn(batch):
    # Implement custom collate function to handle video frames and text
    # This is a placeholder and needs to be implemented based on your specific requirements
    pass


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
