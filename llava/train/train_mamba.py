import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Callable
import warnings

import torch
import transformers
from transformers import TrainingArguments, AutoTokenizer

from llava import VoCoLlamaForVideo, data
from llava.constants import IGNORE_INDEX
from llava.conversation import Conversation, conv_templates
from llava.mm_utils import process_images, tokenizer_image_token

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ModelArguments:
    model_path: str = field()
    num_voco: int = field()
    attn_implementation: str = field(default="sdpa")
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    conv_name: str = field(default="vicuna_video_caption")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)


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
        return data.InternVidDataset(
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
        return data.ActivityNet(
            root_dir=data_args.data_dir,
            split=data_args.data_split,
            labeltype=data_args.dataset.split('-')[-1],
            n_frames=data_args.num_frames,
            fps=data_args.fps,
            frame_shape=(data_args.frame_size, data_args.frame_size),
            transform=image_processor,
            text_preprocess=text_processor,
        )


def process_text(input_dict: dict, tokenizer, conv: Conversation, num_voco: int, gen_prompt=False):
    conv = conv.copy()
    match input_dict:
        case {'caption': caption}:
            if gen_prompt:
                conv.append_message(conv.roles[1], '')
            else:
                conv.append_message(conv.roles[1], caption)
        case {'q': q, 'a': a}:
            conv.append_message(conv.roles[0], q)
            if gen_prompt:
                conv.append_message(conv.roles[1], '')
            else:
                conv.append_message(conv.roles[1], a)
        case {'q': q}:
            conv.append_message(conv.roles[0], q)
            if gen_prompt:
                conv.append_message(conv.roles[1], '')
        case _:
            raise ValueError("Invalid input_dict keys")

    prompt = f"<image>\n{'<voco>' * num_voco}\n{conv.get_prompt()}"
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
    output = {'input_ids': input_ids, 'labels': None}
    if not gen_prompt:
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


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Prepare model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    model = VoCoLlamaForVideo.from_pretrained(
        model_args.model_path,
        attn_implementation=model_args.attn_implementation,
    )
    model.model.vision_tower.load_model()
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     model_args.model_path, None, model_name, init_vision=True,
    #     attn_implementation=model_args.attn_implementation,
    #     deepspeed='zero3'
    # )
    if not hasattr(model.config, 'voco_token_id'):
        model.config.voco_token_id = tokenizer.additional_special_tokens_ids[0]
        model.config.num_voco_tokens = model_args.num_voco

    # Prepare dataset
    conv = conv_templates[model_args.conv_name]
    image_processor = lambda images: process_images(images, model.get_vision_tower().image_processor, model.config)
    text_processor = lambda input_dict: process_text(input_dict, tokenizer, conv, model_args.num_voco)
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
    trainer.save_model(training_args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    # if distributed training, disable logging from subprocesses globally
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        import logging
        logging.disable(logging.ERROR)
    main()
