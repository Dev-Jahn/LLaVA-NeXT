import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator

from llava.mm_utils import process_images
from llava.model_voco import VoCoLlamaForVideo
from llava.train.train_voco_video import ModelArguments, DataArguments, get_dataset, process_text, VideoDataCollator
from llava import conversation

# path = '../checkpoints/voco_llama_ckpt_0807/'
path = '../checkpoints/voco-7b-video/checkpoint-147/'
# path = '../checkpoints/voco-7b-video/checkpoint-294//'
accelerator = Accelerator(mixed_precision='bf16')

model = VoCoLlamaForVideo.from_pretrained(path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
model.config.voco_token_id = tokenizer.additional_special_tokens_ids[0]
model.config.num_voco_tokens = 2
model.get_vision_tower().load_model(device_map='cuda:0')
config = model.config
processor = model.get_vision_tower().image_processor
context_len = tokenizer.model_max_length

model_args = ModelArguments(model_path=path, pretrain_mm_mlp_adapter='', num_voco=2)
data_args = DataArguments(dataset='activitynet-qa', data_dir='/data/datasets/ActivityNet', num_frames=32,
                          data_split='val')

conv = conversation.custom_vicuna_video_qa
image_processor = lambda images: process_images(images, model.get_vision_tower().image_processor, model.config)
text_processor = lambda d: process_text(d, tokenizer, conv, model_args.num_voco)

dataset = get_dataset(data_args, image_processor, text_processor)

collator = VideoDataCollator(tokenizer)

loader = DataLoader(dataset, 4, shuffle=False, num_workers=4, collate_fn=collator)

model, loader = accelerator.prepare(model, loader)

batch = next(iter(loader))

tokenizer.batch_decode(batch['input_ids'][batch['input_ids'] != -200].view(batch['input_ids'].size(0), -1))

with accelerator.autocast(), torch.inference_mode():
    out = model.generate(
        **batch,
        max_length=100,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        # stopping_criteria=stopping_criteria,
    )
print(out)
