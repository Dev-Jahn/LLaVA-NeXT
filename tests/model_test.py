import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from transformers import AutoTokenizer
from accelerate import Accelerator

from llava.model_voco import VoCoLlamaForVideo
from llava.train.train_voco_video import ModelArguments, DataArguments, get_dataset
from llava.mm_utils import process_images
from llava import conversation
from llava.train.train_voco_video import process_text
from llava.train.train_voco_video import VideoDataCollator


def main():
    accelerator = Accelerator()

    path = './checkpoints/voco_llama_ckpt_0807/'
    model_args = ModelArguments(model_path=path, pretrain_mm_mlp_adapter='', num_voco=2)
    data_args = DataArguments(
        dataset='activitynet-captions',
        data_dir='~/data/truenas/video/ActivityNet',
        num_frames=32,
    )

    model = VoCoLlamaForVideo.from_pretrained(
        model_args.model_path,
        attn_implementation=model_args.attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    model.config.voco_token_id = tokenizer.additional_special_tokens_ids[0]
    model.config.num_voco_tokens = 2
    model.get_vision_tower().load_model()
    # model.get_vision_tower().load_model(device_map='cuda:0')
    # model.model.vision_tower = model.model.vision_tower.to(dtype=torch.bfloat16)
    config = model.config
    processor = model.get_vision_tower().image_processor
    context_len = tokenizer.model_max_length

    conv = conversation.custom_vicuna_video_caption
    image_processor = lambda images: process_images(images, model.get_vision_tower().image_processor, model.config)
    text_processor = lambda input_dict: process_text(input_dict, tokenizer, conv, model_args.num_voco)
    dataset = get_dataset(data_args, image_processor, text_processor)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=VideoDataCollator(tokenizer))

    model, loader = accelerator.prepare(model, loader)
    batch = next(iter(loader))
    with accelerator.autocast():
        outputs = model(**batch)
    print(outputs)
    # for k, v in collated.items():
    #     collated[k] = v.cuda()

    # input_ids, labels, attention_mask, videos = collated['input_ids'],collated['labels'], collated['attention_mask'], \
    # collated['videos']

    # with accelerator.autocast():
    #     outputs = model(**collated)

    print('done')


if __name__ == "__main__":
    main()
