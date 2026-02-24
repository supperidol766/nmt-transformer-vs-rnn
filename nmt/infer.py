from __future__ import annotations

import re
from typing import List, Tuple

import torch

from .common.device import try_GPU
from .data.dataset import truncate_pad


def preprocess_sentence(sentence: str) -> List[str]:
    s = sentence.strip().lower()
    s = s.replace("\u202f", " ").replace("\xa0", " ")
    s = re.sub(r"([,.!?;:])", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split(" ") if s else []


@torch.no_grad()
def predict_for_translate(
    net,
    src_sentence: str,
    src_vocab,
    tgt_vocab,
    num_steps: int,
    device: torch.device | None = None,
    save_attention_weights: bool = False,
) -> Tuple[str, list]:
    if device is None:
        device = try_GPU()

    net.eval()
    eos_src = src_vocab["<eos>"]
    eos_tgt = tgt_vocab["<eos>"]
    bos_tgt = tgt_vocab["<bos>"]
    pad_src = src_vocab["<pad>"]
    pad_tgt = tgt_vocab["<pad>"]

    src_tokens = preprocess_sentence(src_sentence)
    src_ids = src_vocab[src_tokens] + [eos_src]
    enc_valid_len = torch.tensor([min(len(src_ids), num_steps)], device=device)
    src_ids = truncate_pad(src_ids, num_steps, pad_src)
    enc_x = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    enc_outputs = net.encoder(enc_x, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)

    dec_x = torch.tensor([[bos_tgt]], dtype=torch.long, device=device)
    output_ids: List[int] = []
    attention_weight_seq = []

    for _ in range(num_steps):
        y, dec_state = net.decoder(dec_x[:, -1:], dec_state)  # (1,1,V)
        pred = int(y[:, -1, :].argmax(dim=-1).item())
        if pred == eos_tgt:
            break
        output_ids.append(pred)
        dec_x = torch.cat([dec_x, torch.tensor([[pred]], device=device)], dim=1)

        if save_attention_weights and hasattr(net.decoder, "attention_weights"):
            attention_weight_seq.append(getattr(net.decoder, "attention_weights"))

    if hasattr(tgt_vocab, "to_tokens"):
        out_tokens = tgt_vocab.to_tokens(output_ids)
    else:
        out_tokens = [tgt_vocab.idx_to_token[i] for i in output_ids if i != pad_tgt]
    return " ".join(out_tokens), attention_weight_seq