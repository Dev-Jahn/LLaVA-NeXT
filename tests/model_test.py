import os, sys
import warnings

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoConfig, AutoTokenizer, AutoModelForCausalLM

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images
from llava.model_voco import VoCoLlamaForCausalLM, VoCoLlamaForVideo
from data.internvid import InternVidDataset
from llava.train.train_voco_video import ModelArguments, DataArguments, get_dataset
from llava import conversation

path = '../checkpoints/voco_llama_ckpt_0807/'

# model = VoCoLlamaForCausalLM.from_pretrained(path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = VoCoLlamaForVideo.from_pretrained(path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
model.config.voco_token_id = tokenizer.additional_special_tokens_ids[0]
model.config.num_voco_tokens = 2
# model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(path)
model.get_vision_tower().load_model(device_map='cuda:0')
model.model.vision_tower = model.model.vision_tower.to(dtype=torch.bfloat16)
config = model.config
processor = model.get_vision_tower().image_processor
context_len = tokenizer.model_max_length

model_args = ModelArguments(model_path=path, pretrain_mm_mlp_adapter='', num_voco=2)
data_args = DataArguments(dataset='internvid', cookie_path='./cookies.txt')

# from llava.mm_utils import process_images
# from llava.train.train_voco_video import process_caption
# conv = conversation.custom_vicuna_video_v1
# image_processor = lambda images: process_images(images, model.get_vision_tower().image_processor, model.config)
# text_processor = lambda text: process_caption(text, tokenizer, conv, model_args.num_voco)
# dataset = get_dataset(data_args, image_processor, text_processor)
#
# batch = [dataset[i] for i in range(4)]
#
# from llava.train.train_voco_video import VideoDataCollator
# collator = VideoDataCollator(tokenizer, pad=True)

# collated = collator(batch)
collated = torch.load('../notebooks/collated.pth')

from accelerate import Accelerator

accelerator = Accelerator(mixed_precision='bf16')

for k, v in collated.items():
    collated[k] = v.cuda()

# input_ids, labels, attention_mask, videos = collated['input_ids'],collated['labels'], collated['attention_mask'], \
# collated['videos']

with accelerator.autocast():
    outputs = model(**collated)

print('done')
