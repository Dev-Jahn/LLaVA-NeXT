import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from transformers import AutoTokenizer
from accelerate import Accelerator

from llava import conversation
from llava.model_voco import VoCoLlamaForVideo
from llava.mm_utils import process_images
from llava.data.utils import ParallelLoaderWrapper
from llava.train.train_voco_video import process_text, VideoDataCollator, ModelArguments, DataArguments, get_dataset


def main():
    accelerator = Accelerator(mixed_precision='bf16')
    path = '../checkpoints/voco_llama_ckpt_0807/'
    model_args = ModelArguments(model_path=path, pretrain_mm_mlp_adapter='', num_voco=2)
    data_args = DataArguments(
        dataset='activitynet-instruct',
        data_dir='/data/datasets/ActivityNet',
        num_frames=32,
    )

    model = VoCoLlamaForVideo.from_pretrained(
        model_args.model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='sdpa',
    ).train()
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    model.config.voco_token_id = tokenizer.additional_special_tokens_ids[0]
    model.config.num_voco_tokens = 2
    model.get_vision_tower().load_model(device_map='auto')

    conv = conversation.custom_vicuna_video_inst
    image_processor = lambda images: process_images(images, model.get_vision_tower().image_processor, model.config)
    text_processor = lambda input_dict: process_text(input_dict, tokenizer, conv, model_args.num_voco)
    dataset = get_dataset(data_args, image_processor, text_processor)
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=VideoDataCollator(tokenizer)
    )
    model, loader = accelerator.prepare(model, loader)
    batch = next(iter(loader))
    with accelerator.autocast():
        outputs = model(**batch)
    print(outputs)
    print('done')


if __name__ == "__main__":
    main()
