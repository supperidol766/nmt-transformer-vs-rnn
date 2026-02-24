from __future__ import annotations

import argparse
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

from nmt.common.device import try_GPU
from nmt.infer import preprocess_sentence
from scripts.entrypoint import load_all


def _trim_ids(seq: Sequence[int], eos_id: int, pad_id: int) -> List[int]:
    out: List[int] = []
    for x in seq:
        xi = int(x)
        if xi == eos_id:
            break
        if xi != pad_id:
            out.append(xi)
    return out


def _ids_to_tokens(vocab, ids: List[int]) -> List[str]:
    if hasattr(vocab, "to_tokens"):
        return vocab.to_tokens(ids)
    return [vocab.idx_to_token[i] for i in ids]


def _tokens_to_text(tokens: List[str]) -> str:
    # 先用简单 join（和你仓库当前 preprocess 兼容）
    return " ".join(tokens)


def _encode_sentence(s: str, src_vocab, num_steps: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = preprocess_sentence(s)
    ids = src_vocab[tokens] + [src_vocab["<eos>"]]
    true_len = min(len(ids), num_steps)
    pad_id = src_vocab["<pad>"]
    if len(ids) < num_steps:
        ids = ids + [pad_id] * (num_steps - len(ids))
    else:
        ids = ids[:num_steps]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    x_len = torch.tensor([true_len], dtype=torch.long, device=device)
    return x, x_len


@torch.no_grad()
def greedy_decode_one(net, src_ids: torch.Tensor, src_len: torch.Tensor, bos_id: int, eos_id: int, max_len: int) -> List[int]:
    device = src_ids.device
    net.eval()
    enc_out = net.encoder(src_ids, src_len)
    state = net.decoder.init_state(enc_out, src_len)

    dec_x = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
    out_ids: List[int] = []
    for _ in range(max_len):
        y, state = net.decoder(dec_x[:, -1:], state)  # (1,1,V)
        next_id = int(y[:, -1, :].argmax(dim=-1).item())
        if next_id == eos_id:
            break
        out_ids.append(next_id)
        dec_x = torch.cat([dec_x, torch.tensor([[next_id]], device=device)], dim=1)
    return out_ids


def _repeat_enc_output_for_beam(enc_out, beam_size: int):
    """
    Transformer encoder output is usually Tensor (B,T,H).
    Seq2Seq encoder output is usually tuple (enc_outputs, hidden_state).
    """
    if isinstance(enc_out, torch.Tensor):
        if enc_out.size(0) == 1:
            return enc_out.repeat(beam_size, 1, 1)
        if enc_out.dim() >= 2 and enc_out.size(1) == 1:
            return enc_out.repeat(1, beam_size, 1)
        raise ValueError(f"Cannot infer batch dimension for encoder tensor shape={tuple(enc_out.shape)}")

    if isinstance(enc_out, (tuple, list)) and len(enc_out) >= 2:
        out, st = enc_out[0], enc_out[1]
        # seq2seq enc out often (T,B,H), hidden often (L,B,H)
        if isinstance(out, torch.Tensor):
            if out.dim() >= 2 and out.size(1) == 1:
                out = out.repeat(1, beam_size, 1)
            elif out.size(0) == 1:
                out = out.repeat(beam_size, 1, 1)
            else:
                raise ValueError(f"Cannot infer batch dim for seq2seq enc_outputs shape={tuple(out.shape)}")
        if isinstance(st, torch.Tensor):
            if st.dim() >= 2 and st.size(1) == 1:
                st = st.repeat(1, beam_size, 1)
            elif st.size(0) == 1:
                st = st.repeat(beam_size, 1, 1)
            else:
                raise ValueError(f"Cannot infer batch dim for seq2seq hidden shape={tuple(st.shape)}")
        if isinstance(enc_out, tuple):
            return (out, st)
        new_state = list(enc_out)
        new_state[0], new_state[1] = out, st
        return type(enc_out)(new_state)

    raise TypeError(f"Unsupported encoder output type for beam repeat: {type(enc_out)}")


def _reindex_batch_tensor(x: torch.Tensor, beam_idx: torch.Tensor) -> torch.Tensor:
    """
    Reorder a tensor by beam indices along its batch dimension.
    Batch dimension can be dim0 (B,...) or dim1 (T,B,...)/(L,B,...).
    """
    if x.dim() == 0:
        return x
    b = beam_idx.numel()
    if x.size(0) == b:
        return x.index_select(0, beam_idx)
    if x.dim() >= 2 and x.size(1) == b:
        return x.index_select(1, beam_idx)
    for d in range(min(3, x.dim())):
        if x.size(d) == b:
            return x.index_select(d, beam_idx)
    return x  # unknown tensor layout: leave unchanged


def _reorder_decoder_state(state, beam_idx: torch.Tensor):
    """
    IMPORTANT FIX:
    For seq2seq beam search, reorder BOTH hidden state and encoder outputs.
    Previously only hidden/valid_lens were reordered, causing hidden-memory mismatch.
    """
    if isinstance(state, torch.Tensor):
        return _reindex_batch_tensor(state, beam_idx)

    if isinstance(state, tuple):
        items = list(state)
        for i, item in enumerate(items):
            if isinstance(item, torch.Tensor):
                items[i] = _reindex_batch_tensor(item, beam_idx)
            elif isinstance(item, (tuple, list)):
                nested = []
                for v in item:
                    nested.append(_reindex_batch_tensor(v, beam_idx) if isinstance(v, torch.Tensor) else v)
                items[i] = type(item)(nested) if isinstance(item, tuple) else nested
        return tuple(items)

    if isinstance(state, list):
        out = []
        for item in state:
            if isinstance(item, torch.Tensor):
                out.append(_reindex_batch_tensor(item, beam_idx))
            elif isinstance(item, (tuple, list)):
                nested = []
                for v in item:
                    nested.append(_reindex_batch_tensor(v, beam_idx) if isinstance(v, torch.Tensor) else v)
                out.append(type(item)(nested) if isinstance(item, tuple) else nested)
            else:
                out.append(item)
        return out

    return state


@torch.no_grad()
def beam_search_one(
    net,
    src_ids: torch.Tensor,
    src_len: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
    beam_size: int,
    len_penalty: float,
) -> List[int]:
    """
    Beam search for a single sentence. Works for both seq2seq and transformer.
    """
    device = src_ids.device
    net.eval()

    enc_out = net.encoder(src_ids, src_len)
    enc_out_rep = _repeat_enc_output_for_beam(enc_out, beam_size)
    src_len_rep = src_len.repeat(beam_size)
    state = net.decoder.init_state(enc_out_rep, src_len_rep)

    tokens = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)
    logp = torch.zeros((beam_size,), dtype=torch.float32, device=device)
    ended = torch.zeros((beam_size,), dtype=torch.bool, device=device)

    for _ in range(max_len):
        y, state = net.decoder(tokens[:, -1:], state)  # (B,1,V)
        next_logp = F.log_softmax(y[:, -1, :], dim=-1)  # (B,V)

        # 已结束 beam 只能继续 eos
        if ended.any():
            next_logp = next_logp.masked_fill(ended.unsqueeze(1), float("-inf"))
            next_logp[ended, eos_id] = 0.0

        vocab_size = next_logp.size(-1)
        cand_logp = (logp.unsqueeze(1) + next_logp).reshape(-1)
        topk_logp, topk_idx = torch.topk(cand_logp, k=beam_size)

        beam_idx = topk_idx // vocab_size
        tok_idx = topk_idx % vocab_size

        tokens = tokens.index_select(0, beam_idx)
        tokens = torch.cat([tokens, tok_idx.unsqueeze(1)], dim=1)
        logp = topk_logp
        ended = ended.index_select(0, beam_idx) | tok_idx.eq(eos_id)

        # 关键修复：state 必须按 beam_idx 重排（包含 seq2seq 的 enc_outputs）
        state = _reorder_decoder_state(state, beam_idx)

        if bool(ended.all()):
            break

    best_score = None
    best_seq: List[int] = []
    for i in range(beam_size):
        seq = tokens[i, 1:].tolist()  # drop BOS
        if eos_id in seq:
            effective_len = seq.index(eos_id) + 1
        else:
            effective_len = len(seq)
        denom = (effective_len ** len_penalty) if len_penalty > 0 else 1.0
        score = float(logp[i].item()) / max(denom, 1e-6)
        if best_score is None or score > best_score:
            best_score = score
            best_seq = _trim_ids(seq, eos_id=eos_id, pad_id=-999999)
    return best_seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
    ap.add_argument("--sentence", type=str, default=None, help="Single source sentence.")
    ap.add_argument("--input_file", type=str, default=None, help="One source sentence per line.")
    ap.add_argument("--output_file", type=str, default=None, help="Output file path for predictions.")
    ap.add_argument("--decode", type=str, default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--beam_size", type=int, default=4)
    ap.add_argument("--len_penalty", type=float, default=0.6)
    ap.add_argument("--max_len", type=int, default=None, help="Override decode max length.")
    args = ap.parse_args()

    if (args.sentence is None) == (args.input_file is None):
        raise ValueError("Provide exactly one of --sentence or --input_file.")

    device = try_GPU() if args.device == "auto" else torch.device(args.device)
    net, src_vocab, tgt_vocab, _, cfg = load_all(args.config, device=device)

    num_steps = int(cfg["data"]["num_steps"])
    max_len = int(args.max_len) if args.max_len is not None else num_steps

    bos_id = tgt_vocab["<bos>"]
    eos_id = tgt_vocab["<eos>"]
    pad_id = tgt_vocab["<pad>"]

    if args.sentence is not None:
        inputs = [args.sentence.strip()]
    else:
        with open(args.input_file, "r", encoding="utf-8") as f:
            inputs = [line.strip() for line in f if line.strip()]

    preds: List[str] = []
    for s in inputs:
        x, x_len = _encode_sentence(s, src_vocab, num_steps=num_steps, device=device)

        if args.decode == "greedy":
            pred_ids = greedy_decode_one(net, x, x_len, bos_id, eos_id, max_len=max_len)
        else:
            pred_ids = beam_search_one(
                net, x, x_len, bos_id, eos_id,
                max_len=max_len,
                beam_size=int(args.beam_size),
                len_penalty=float(args.len_penalty),
            )

        pred_ids = _trim_ids(pred_ids, eos_id=eos_id, pad_id=pad_id)
        pred_tokens = _ids_to_tokens(tgt_vocab, pred_ids)
        preds.append(_tokens_to_text(pred_tokens))

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(p + "\n")
    else:
        for p in preds:
            print(p)


if __name__ == "__main__":
    main()