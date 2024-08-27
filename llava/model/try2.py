from abc import abstractmethod

import torch

from llava.constants import IGNORE_INDEX
from llava.model_voco.llava_arch import LlavaMetaForCausalLM


class VoCoMetaForVideo(LlavaMetaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.voco_token_id = config.voco_token_id  # Assume this is defined in the config

    @abstractmethod
    def get_model(self):
        ...

    def encode_video(self, video_frames):
        # Assume video_frames is a tensor of shape (batch_size, num_frames, num_patches, hidden_size)
        batch_size, num_frames, num_patches, hidden_size = video_frames.shape

        # Process each frame independently
        video_features = []
        for frame in range(num_frames):
            frame_features = self.get_model().get_vision_tower()(video_frames[:, frame])
            frame_features = self.get_model().mm_projector(frame_features)

            # Add <voco> token representation
            voco_embed = self.get_model().embed_tokens(torch.tensor([self.voco_token_id], device=frame_features.device))
            frame_features = torch.cat([frame_features, voco_embed], dim=1)

            # Get only the <voco> token representation
            voco_representation = frame_features[:, -1]
            video_features.append(voco_representation)

        # Stack all <voco> representations
        video_features = torch.stack(video_features, dim=1)  # Shape: (batch_size, num_frames, hidden_size)
        return video_features

    def prepare_inputs_labels_for_video(
            self, input_ids, position_ids, attention_mask, past_key_values, labels,
            videos, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or videos is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        video_features = self.encode_video(videos)  # Shape: (batch_size, num_frames, hidden_size)

        new_input_embeds = []
        new_labels = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_video_features = video_features[batch_idx]
            num_voco_tokens = cur_video_features.shape[0]

            # Embed text tokens
            cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)

            # Concatenate video features (voco tokens) with text embeddings
            cur_input_embeds = torch.cat([cur_video_features, cur_input_embeds], dim=0)

            new_input_embeds.append(cur_input_embeds)

            # Adjust labels
            cur_labels = labels[batch_idx]
            new_labels.append(torch.cat([
                torch.full((num_voco_tokens,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype),
                cur_labels
            ]))

        # Combine and pad the sequences
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        hidden_size = new_input_embeds[0].shape[-1]

        new_input_embeds_padded = torch.zeros((batch_size, max_len, hidden_size), dtype=new_input_embeds[0].dtype,
                                              device=new_input_embeds[0].device)
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
        attention_mask_padded = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype,
                                            device=attention_mask.device)

        for i, (cur_embed, cur_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_embed.shape[0]
            new_input_embeds_padded[i, :cur_len] = cur_embed
            new_labels_padded[i, :cur_len] = cur_labels
            attention_mask_padded[i, :cur_len] = 1

        # Adjust position_ids to account for added voco tokens
        if position_ids is not None:
            position_ids = position_ids + num_voco_tokens

        return None, position_ids, attention_mask_padded, past_key_values, new_input_embeds_padded, new_labels_padded

    def forward(self, *args, **kwargs):
        videos = kwargs.pop("videos", None)

        if videos is not None:
            inputs_embeds = kwargs.get("inputs_embeds")
            input_ids = kwargs.get("input_ids")
            attention_mask = kwargs.get("attention_mask")
            labels = kwargs.get("labels")
            position_ids = kwargs.get("position_ids")
            past_key_values = kwargs.get("past_key_values")
            image_sizes = kwargs.get("image_sizes")

            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_video(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                videos,
                image_sizes,
            )

            kwargs.update({
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "inputs_embeds": inputs_embeds,
                "labels": labels
            })

        return super().forward(*args, **kwargs)
