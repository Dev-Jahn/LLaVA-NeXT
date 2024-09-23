import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Callable
import warnings

import torch
import transformers
from transformers import TrainingArguments

from data.internvid import InternVidDataset
from llava.constants import IGNORE_INDEX
from llava.conversation import Conversation, conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.train.train import safe_save_model_for_hf_trainer, get_peft_state_maybe_zero_3, \
    get_peft_state_non_lora_maybe_zero_3
from llava.train.llava_trainer import LLaVATrainer

local_rank = None
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ModelArguments:
    model_path: str = field()
    num_voco: int = field()
    attn_implementation: str = field(default="sdpa")
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    conv_name: str = field(default='custom_vicuna_video')
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    # mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    # mm_projector_type: Optional[str] = field(default='linear')
    # mm_use_im_start_end: bool = field(default=False)
    # mm_use_im_patch_token: bool = field(default=True)
    # mm_patch_merge_type: Optional[str] = field(default='flat')
    # mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    dataset: str = field()
    cookie_path: Optional[str] = field(default=None)
    data_dir: Optional[str] = field(default=None, metadata={"help": "Path to the data directory."})
    fps: Optional[int] = None
    num_frames: int = None
    frame_size: int = 336
    hwaccel: Optional[str] = field(default=None)


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


def get_dataset(data_args: DataArguments, image_processor: Callable, text_processor: Callable):
    if data_args.dataset == "internvid":
        return InternVidDataset(
            cache_dir=data_args.data_dir,
            fps=data_args.fps,
            max_frames=data_args.num_frames,
            frame_size=data_args.frame_size,
            hwaccel=data_args.hwaccel,
            transform=image_processor,
            text_preprocess=text_processor,
            cookie_path=data_args.cookie_path,
        )


def process_caption(caption: str, tokenizer, conv: Conversation, num_voco: int) -> Dict[str, torch.Tensor]:
    conv = conv.copy()
    conv.append_message(
        conv.roles[0],
        "Describe the main visual content or key elements you observe in each video clip in a single lowercase sentence."
    )
    conv.append_message(
        conv.roles[1],
        caption
    )
    prompt = f"<image>\n{'<voco>' * num_voco}\n{conv.get_prompt()}"
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
    targets = torch.full_like(input_ids, IGNORE_INDEX)
    cap_len = len(tokenizer(caption, return_attention_mask=False)['input_ids'])
    targets[-cap_len:] = input_ids[-cap_len:]
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


@dataclass
class VideoDataCollator:
    """
    Does not pad tensors to avoid redundant jobs
    """
    tokenizer: transformers.PreTrainedTokenizer
    pad: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if self.pad:
            input_ids, labels, videos = [[instance[key] for instance in instances] for key in
                                         ['input_ids', 'labels', 'frames']]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                     batch_first=True,
                                                     padding_value=IGNORE_INDEX)
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            labels = labels[:, :self.tokenizer.model_max_length]
            # TODO: Currently not considering video batch with jagged frames
            batch = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                videos=torch.stack(videos)
            )
            return batch
        else:
            input_ids, labels, videos = [[instance[key] for instance in instances] for key in
                                         ['input_ids', 'labels', 'frames']]
            batch = dict(
                input_ids=input_ids,
                labels=labels,
                videos=videos
            )
            return batch


def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Prepare model and tokenizer
    model_name = get_model_name_from_path(model_args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_args.model_path, None, model_name, init_vision=True,
        attn_implementation=model_args.attn_implementation,
        deepspeed='zero3'
    )
    # Prepare dataset
    conv = conv_templates[model_args.conv_name]
    image_processor = lambda images: process_images(images, model.get_vision_tower().image_processor, model.config)
    text_processor = lambda text: process_caption(text, tokenizer, conv, model_args.num_voco)
    dataset = get_dataset(data_args, image_processor, text_processor)

    # Prepare trainer
    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=VideoDataCollator(tokenizer),
    )

    # Start training
    trainer.train()
    trainer.save_state(trainer)

    # Save the final model
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(
                training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(
                training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    # if distributed training, disable logging from subprocesses globally
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        import logging

        logging.disable(logging.ERROR)
    main()
