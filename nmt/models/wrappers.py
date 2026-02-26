from __future__ import annotations
import torch
from torch import nn
class EncoderDecoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, enc_x: torch.Tensor, dec_x: torch.Tensor, enc_valid_lens: torch.Tensor):
        enc_outputs = self.encoder(enc_x, enc_valid_lens)
        dec_state = self.decoder.init_state(enc_outputs, enc_valid_lens)
    
        return self.decoder(dec_x, dec_state)

