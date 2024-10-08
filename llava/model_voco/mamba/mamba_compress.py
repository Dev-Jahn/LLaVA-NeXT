from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import Mamba2PreTrainedModel
from transformers.models.mamba2.modeling_mamba2 import Mamba2Block, Mamba2RMSNorm, Mamba2Cache, Mamba2Output


class Mamba2Encoder(Mamba2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.layers = nn.ModuleList([Mamba2Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            inputs_embeds: Optional[torch.LongTensor] = None,
            cache_params: Optional[Mamba2Cache] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[Tuple, Mamba2Output]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            raise ValueError("You have to specify `inputs_embeds`")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = Mamba2Cache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, cache_params, cache_position, attention_mask
                )
            else:
                hidden_states = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return Mamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


class Mamba2ForVideoCompression(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vison_encoder =
        for param in self.clip_encoder.parameters():
            param.requires_grad = False

        self.spatial_mamba = Mamba2ForCompression(config)
        self.temporal_mamba = Mamba2ForCompression(config)

    def forward(self, frames):
        batch_size, num_frames, c, h, w = frames.shape

        # Encode frames with CLIP
        encoded_frames = self.clip_encoder(frames.view(-1, c, h, w))
        encoded_frames = encoded_frames.view(batch_size, num_frames, -1, encoded_frames.shape[-1])

        # Step 2: Spatial Mamba
        spatial_hidden = []
        for frame in encoded_frames:
            spatial_output = self.spatial_mamba(inputs_embeds=frame).last_hidden_state[:, -1, :]
            spatial_hidden.append(spatial_output)
        spatial_hidden = torch.stack(spatial_hidden, dim=1)

        # Step 3: Temporal Mamba on spatial hidden states
        temporal_output_3 = self.temporal_mamba(inputs_embeds=spatial_hidden).last_hidden_state[:, -1, :]

        # Step 4: Temporal Mamba on encoded frames
        temporal_hidden = []
        for patch in encoded_frames.transpose(1, 2):
            temporal_output = self.temporal_mamba(inputs_embeds=patch).last_hidden_state[:, -1, :]
            temporal_hidden.append(temporal_output)
        temporal_hidden = torch.stack(temporal_hidden, dim=1)

        # Step 5: Spatial Mamba on temporal hidden states
        spatial_output_5 = self.spatial_mamba(inputs_embeds=temporal_hidden).last_hidden_state[:, -1, :]

        return temporal_output_3, spatial_output_5