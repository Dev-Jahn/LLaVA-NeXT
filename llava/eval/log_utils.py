from PIL import Image
import numpy as np
import torch
import wandb


def replace_image_index(input_ids, tokenizer):
    input_ids = input_ids.detach().clone().cpu()
    indices = (input_ids == -200).nonzero(as_tuple=True)[1]
    img_tokens = tokenizer.encode('<image>', return_tensors='pt')[0, 1:]
    new_input_ids = []
    for i, ids in zip(indices, input_ids):
        new_input_ids.append(torch.cat((ids[:i], img_tokens, ids[i + 1:]), dim=0))
    return torch.stack(new_input_ids)


def denormalize_video(video_tensor, image_mean, image_std):
    video_tensor = video_tensor.clone().cpu().detach()
    mean = torch.tensor(image_mean).view(1, 1, 3, 1, 1)
    std = torch.tensor(image_std).view(1, 1, 3, 1, 1)
    video_np = (video_tensor.cpu() * std + mean).numpy()
    return (video_np * 255).astype(np.uint8)


def video2wandb(table: wandb.Table, video_tensor, processor, **kwargs):
    mean, std = processor.image_mean, processor.image_std
    kwargs.update({'video': denormalize_video(video_tensor, mean, std)})
    table.add_data([kwargs.get(col) for col in table.columns])
    return table
