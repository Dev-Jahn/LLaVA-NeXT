from typing import List, Optional, Tuple, Union
import torch
from transformers.utils import PushToHubMixin


class VideoTensorTokenizer(PushToHubMixin):
    def __init__(self, spatial_compression: int = 2, temporal_compression: int = 2):
        super().__init__()
        self.spatial_compression = spatial_compression
        self.temporal_compression = temporal_compression

        # Define special tokens
        self.special_tokens = {
            "<patch>": -200,    # Placeholder for vision encoded features
            "<pad>": 0,         # Pad token for different length sequences
            "<bof>": 1,         # Beginning of a frame
            "<hsep>": 2,        # Horizontal separator
            "<vsep>": 3,        # Vertical separator
            "<eof>": 4,         # End of a frame
            "<scomp>": 5,       # Spatial compression
            "<bot>": 6,         # Beginning of time
            "<eot>": 7,         # End of time
            "<tcomp>": 8,       # Temporal compression
            "<mask>": 9,        # Reserved for MLM
        }

        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

    def _tokenize(self, video_tensor: torch.Tensor) -> Tuple[List[int], List[int]]:
        batch, frames, channels, height, width = video_tensor.shape

        spatial_tokens = []
        temporal_tokens = []

        # Spatial tokenization
        for _ in range(frames):
            spatial_tokens.extend([self.special_tokens["<bof>"]])
            for _ in range(height):
                for _ in range(width):
                    spatial_tokens.extend([self.special_tokens["<patch>"]])
                if _ < height - 1:
                    spatial_tokens.extend([self.special_tokens["<hsep>"]])
            spatial_tokens.extend([self.special_tokens["<eof>"]])
            spatial_tokens.extend([self.special_tokens["<scomp>"]] * self.spatial_compression)

        # Temporal tokenization
        temporal_tokens.extend([self.special_tokens["<bot>"]])
        for _ in range(frames):
            for _ in range(height * width):
                temporal_tokens.extend([self.special_tokens["<patch>"]])
        temporal_tokens.extend([self.special_tokens["<eot>"]])
        temporal_tokens.extend([self.special_tokens["<tcomp>"]] * self.temporal_compression)

        return spatial_tokens, temporal_tokens

    def encode(
        self,
        video_tensor: torch.Tensor,
        return_tensors: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_tokens, temporal_tokens = self._tokenize(video_tensor)

        if return_tensors == "pt":
            return torch.tensor(spatial_tokens, dtype=torch.long), torch.tensor(temporal_tokens, dtype=torch.long)
        else:
            return spatial_tokens, temporal_tokens

    def decode(self, token_ids: Union[int, List[int]]) -> str:
        return " ".join(self.id_to_token.get(token_id, f"<{token_id}>") for token_id in token_ids)

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        else:
            return [1 if token in self.all_special_ids else 0 for token in
                    self.build_inputs_with_special_tokens(token_ids_0, token_ids_1)]

    @property
    def vocab_size(self) -> int:
        return len(self.special_tokens)

    def get_vocab(self):
        return dict(self.special_tokens)

    def _convert_token_to_id(self, token: str) -> int:
        return self.special_tokens.get(token, self.special_tokens["<pad>"])

    def _convert_id_to_token(self, index: int) -> str:
        return self.id_to_token.get(index, f"<{index}>")

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # This tokenizer doesn't have a vocabulary file, so just return an empty tuple
        return ()
