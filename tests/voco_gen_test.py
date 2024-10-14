import os, sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from accelerate import Accelerator
import numpy as np
import wandb
from PIL import Image

sys.path.append(os.path.dirname(os.getcwd()))

from llava.model_voco.language_model.llava_voco_llama import VoCoConfig
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, KeywordsStoppingCriteria2
from llava.model_voco import VoCoLlamaForCausalLM, VoCoLlamaForVideo
from llava.data.internvid import InternVidDataset
from llava.train.train_voco_video import ModelArguments, DataArguments, get_dataset
from llava import conversation
from llava.conversation import SeparatorStyle

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# path = 'liuhaotian/llava-v1.5-7b'
# path = '../checkpoints/voco_llama_ckpt_0807'
path = '../checkpoints/voco_llama_ckpt_freeze_mlp/'
# path = '../checkpoints/voco-7b-finetune-1token/'
# path = '../checkpoints/voco-7b-finetune-4token/'

accelerator = Accelerator()
device = accelerator.device
model = VoCoLlamaForCausalLM.from_pretrained(
    path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
# override if incompatible
if not hasattr(model.config, 'voco_token_id'):
    model.config.voco_token_id = tokenizer.additional_special_tokens_ids[0]
if not hasattr(model.config, 'num_voco_tokens'):
    model.config.num_voco_tokens = 2
model.get_vision_tower().load_model(device_map='auto')
model.get_vision_tower().eval().to(device)
config = model.config
processor = model.get_vision_tower().image_processor
context_len = tokenizer.model_max_length

import requests
from io import BytesIO

url = 'https://plus.unsplash.com/premium_photo-1664474619075-644dd191935f?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8JTIzaW1hZ2V8ZW58MHx8MHx8fDA%3D'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img

from llava.mm_utils import process_images
from llava.train.train_voco_video import process_text

conv = conversation.conv_vicuna_v1
image_processor = lambda images: process_images(images, model.get_vision_tower().image_processor, model.config)
text_processor = lambda d: process_text(d, tokenizer, conv, model.config.num_voco_tokens, gen_prompt=True)

# dataset.labels[0]['q'] = 'What is described in this video?'


batch = text_processor({'q': 'What is described in this image?'})
batch.pop('labels')
batch['inputs'] = batch.pop('input_ids').detach().clone().unsqueeze(0).to(device)
batch['attention_mask'] = batch['inputs'].clone().ne(tokenizer.pad_token_id).to(device)
batch['images'] = image_processor([img]).to(device)

from llava.eval.log_utils import replace_image_index

tokenizer.batch_decode(replace_image_index(batch['inputs'], tokenizer), skip_special_tokens=False)

# Single sample test
with accelerator.autocast(), torch.no_grad():
    keywords = [conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2]
    keywords += ['.', "'", ","]
    stopping_criteria = KeywordsStoppingCriteria2(keywords, tokenizer)
    out = model.generate(
        **batch,
        max_new_tokens=64,
        do_sample=False,
        # top_p=1,
        # temperature=0.1,
        # do_sample=True,
        # top_p=0.9,
        # top_k=50,
        # num_beams=4,
        # no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=[stopping_criteria],
    ).squeeze()
    print(tokenizer.decode(out))
