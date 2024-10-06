from PIL import Image
import numpy as np
import torch


def replace_image_index(input_ids, tokenizer):
    input_ids = input_ids.detach().clone().cpu()
    indices = (input_ids == -200).nonzero(as_tuple=True)[1]
    img_tokens = tokenizer.encode('<image>')[1:]
    new_input_ids = []
    for i, ids in zip(indices, input_ids):
        new_input_ids.append(torch.cat((ids[:i], img_tokens, ids[i + 1:]), dim=0))
    return torch.stack(new_input_ids)


def denormalize_video(video_tensor, image_mean, image_std):
    mean = torch.tensor(image_mean).view(1, 1, 3, 1, 1)
    std = torch.tensor(image_std).view(1, 1, 3, 1, 1)
    video_np = (video_tensor.cpu() * std + mean).numpy()
    return (video_np * 255).astype(np.uint8)


video_tensor = batch['videos'].clone().cpu().detach()  # shape: [4, 32, 3, 336, 336]
video_nps = denormalize_video(video_tensor, ip.image_mean, ip.image_std)
