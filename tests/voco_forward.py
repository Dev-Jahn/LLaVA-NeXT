import os, sys
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from torch.utils.data import DataLoader

from llava.train.train_voco import make_supervised_data_module, DataArguments

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from transformers import AutoTokenizer
from accelerate import Accelerator
from PIL import Image

sys.path.append(os.path.dirname(os.getcwd()))

from llava import conversation
from llava.mm_utils import KeywordsStoppingCriteria2, process_images
from llava.model_voco import VoCoLlamaForCausalLM
from llava.conversation import SeparatorStyle
from llava.train.train_voco_video import process_text
from llava.eval.log_utils import replace_image_index

# path = 'liuhaotian/llava-v1.5-7b'
# path = '../checkpoints/voco_llama_ckpt_0807'
path = '../checkpoints/voco_llama_ckpt_freeze_mlp/'
# path = '../checkpoints/voco-7b-finetune-1token/'
# path = '../checkpoints/voco-7b-finetune-4token/'

accelerator = Accelerator(mixed_precision='bf16')
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

ip = model.get_vision_tower().image_processor

data_args = DataArguments(
    data_path='../playground/data/llava_v1_5_mix665k.json',
    lazy_preprocess=True,
    image_folder='../playground/data',
    image_aspect_ratio='pad',
)
data_args.image_processor = ip
from llava import conversation as conversation_lib

conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args,
                                          num_voco=model.config.num_voco_tokens)
loader = DataLoader(
    data_module['train_dataset'],
    8,
    shuffle=False,
    num_workers=4,
    collate_fn=data_module['data_collator']
)

model, loader = accelerator.prepare(model, loader)

batch = next(iter(loader))

print(tokenizer.batch_decode(replace_image_index(batch['input_ids'], tokenizer),
                             skip_special_tokens=False))

# Single sample test
with accelerator.autocast(), torch.no_grad():
    out = model(**batch)
    print(tokenizer.decode(out))
