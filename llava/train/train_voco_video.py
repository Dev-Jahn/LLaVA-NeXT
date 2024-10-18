import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Callable, Union, Any
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoTokenizer

from llava.constants import IGNORE_INDEX
from llava.conversation import Conversation, conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.train.train import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, TrainingArguments, \
    safe_save_model_for_hf_trainer
from llava.train.llava_trainer import LLaVATrainer
from llava.model_voco import VoCoLlamaForVideo
from llava import data as custom_data
from llava.data.utils import ParallelLoaderWrapper
from llava.utils import rank0_print

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ModelArguments:
    model_path: str = field()
    num_voco: int = field()
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    conv_name: str = field(default="vicuna_video_caption")
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
    data_split: str = field(default="train")
    cookie_path: Optional[str] = field(default=None)
    data_dir: Optional[str] = field(default=None, metadata={"help": "Path to the data directory."})
    fps: Optional[int] = None
    num_frames: int = None
    frame_size: int = 336
    hwaccel: Optional[str] = field(default=None)
    dataset_debug: bool = field(default=False)
    video_backend: str = field(default="pyav")


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    ckpt_root: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    model_max_length: int = field(default=2048, metadata={"help": "Maximum seq length."})

    attn_implementation: str = field(default="sdpa")

    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)


def get_dataset(data_args: DataArguments, image_processor: Callable, text_processor: Callable):
    if data_args.dataset == "internvid":
        return custom_data.InternVidDataset(
            cache_dir=data_args.data_dir,
            fps=data_args.fps,
            max_frames=data_args.num_frames,
            frame_size=data_args.frame_size,
            hwaccel=data_args.hwaccel,
            transform=image_processor,
            text_preprocess=text_processor,
            cookie_path=data_args.cookie_path,
        )
    elif 'activitynet' in data_args.dataset:
        return custom_data.ActivityNet(
            root_dir=data_args.data_dir,
            split=data_args.data_split,
            labeltype=data_args.dataset.split('-')[-1],
            n_frames=data_args.num_frames,
            fps=data_args.fps,
            frame_shape=(data_args.frame_size, data_args.frame_size),
            transform=image_processor,
            text_preprocess=text_processor,
            debug=data_args.dataset_debug,
            video_backend=data_args.video_backend,
        )


def process_text(input_dict: dict, tokenizer, conv: Conversation, num_voco: int, generation=False):
    conv = conv.copy()
    match input_dict:
        case {'caption': caption}:
            if generation:
                conv.append_message(conv.roles[1], '')
            else:
                conv.append_message(conv.roles[1], caption)
        case {'q': q, 'a': a}:
            conv.append_message(conv.roles[0], q)
            if generation:
                conv.append_message(conv.roles[1], '')
            else:
                conv.append_message(conv.roles[1], a)
        case {'q': q}:
            conv.append_message(conv.roles[0], q)
            if generation:
                conv.append_message(conv.roles[1], '')
        case _:
            raise ValueError("Invalid input_dict keys")

    prompt = f"<image>\n{'<voco>' * num_voco}\n{conv.get_prompt()}"
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
    output = {'input_ids': input_ids, 'labels': None}
    if not generation:
        targets = torch.full_like(input_ids, IGNORE_INDEX)
        if 'caption' in input_dict:
            cap_len = len(tokenizer(input_dict['caption'], return_attention_mask=False)['input_ids'])
            targets[-cap_len:] = input_ids[-cap_len:]
        elif 'q' in input_dict and 'a' in input_dict:
            a_len = len(tokenizer(input_dict['a'], return_attention_mask=False)['input_ids'])
            targets[-a_len:] = input_ids[-a_len:]
        output['labels'] = targets
    return output


@dataclass
class VideoDataCollator:
    """
    Does not pad tensors to avoid redundant jobs
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances] if instances[0].get('labels') is not None else None
        videos = [instance['video'] for instance in instances] if instances[0].get('video') is not None else None
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX) if labels is not None else None
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length] if labels is not None else None
        batch = {'input_ids': input_ids, 'attention_mask': input_ids.ne(self.tokenizer.pad_token_id)}
        if labels is not None:
            batch['labels'] = labels
        if videos is not None:
            # TODO: Currently not considering video batch with differenct number of frames
            batch['videos'] = torch.stack(videos)
        return batch


class LLaVATrainerWrapper(LLaVATrainer):
    def get_train_dataloader(self) -> DataLoader:
        return ParallelLoaderWrapper(super().get_train_dataloader())

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        return ParallelLoaderWrapper(super().get_eval_dataloader(eval_dataset))

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return ParallelLoaderWrapper(super().get_test_dataloader(test_dataset))

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        # Debug purpose only
        if 7 < self.state.global_step < 914:
            logging.info(f'rank: {self.args.local_rank} | step: {self.state.global_step} | skipped')
            return torch.tensor(torch.nan)
        else:
            logging.info(f'rank: {self.args.local_rank} | step: {self.state.global_step} | step start')
            out = super().training_step(model, inputs)
            logging.info(f'rank: {self.args.local_rank} | step: {self.state.global_step} | step end')
            return out


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Prepare model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    model = VoCoLlamaForVideo.from_pretrained(
        model_args.model_path,
        attn_implementation=training_args.attn_implementation,
    )
    model.model.vision_tower.load_model()
    if not hasattr(model.config, 'voco_token_id'):
        model.config.voco_token_id = tokenizer.additional_special_tokens_ids[0]
        model.config.num_voco_tokens = model_args.num_voco

    model.train()
    model.get_vision_tower().eval()
    model.get_vision_tower().requires_grad_(False)
    model.model.mm_projector.eval()
    model.model.mm_projector.requires_grad_(False)

    # Prepare dataset
    conv = conv_templates[model_args.conv_name]
    image_processor = lambda images: process_images(images, model.get_vision_tower().image_processor, model.config)
    text_processor = lambda input_dict: process_text(input_dict, tokenizer, conv, model_args.num_voco)
    dataset = get_dataset(data_args, image_processor, text_processor)

    # Prepare trainer
    trainer = LLaVATrainerWrapper(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=VideoDataCollator(tokenizer),
    )

    # Start training
    if training_args.resume_from_checkpoint:
        checkpoint = os.path.join(training_args.output_dir, training_args.resume_from_checkpoint)
    else:
        checkpoint = None
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    # if distributed training, disable logging from subprocesses globally
    # if int(os.environ.get("LOCAL_RANK", 0)) != 0:
    #     import logging

    # logging.disable(logging.ERROR)
    main()
