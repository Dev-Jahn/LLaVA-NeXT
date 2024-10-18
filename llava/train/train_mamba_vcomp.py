import torch
import torch.nn.functional as F

from transformers import Trainer


def kl_div(p, q):
    return torch.sum(p * torch.log(p / q), dim=-1)


def js_div(p, q):
    m = 0.5 * (p + q)  # Compute the average distribution
    return 0.5 * (kl_div(p, m) + kl_div(q, m))


class VideoCompressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        frames = inputs["frames"]
        ts_out, st_out = model(frames)

        # Apply softmax to get probability distributions
        ts_probs = F.softmax(ts_out, dim=-1)
        st_probs = F.softmax(st_out, dim=-1)

        # Compute JS divergence
        loss = js_div(ts_probs, st_probs).mean()

        return (loss, (ts_out, st_out)) if return_outputs else loss
