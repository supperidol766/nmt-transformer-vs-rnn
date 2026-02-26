from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

from nmt.data.dataset import tokenize
from nmt.data.vocab import Vocab
from nmt.models.Transformer import TransformerEncoder, TransformerDecoder
from nmt.models.seq2seq import Seq2seqEncoder, Seq2seqAttentionDecoder
from nmt.models.wrappers import EncoderDecoder


@dataclass
class LoadedState:
    epoch: int = 0
    step: int = 0
    best_bleu: float = 0.0
    optimizer_state: Optional[dict] = None
    scheduler_state: Optional[dict] = None


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _vocab_to_json(vocab: Vocab) -> dict:
    return {"idx_to_token": list(vocab.idx_to_token)}


def _vocab_from_json(obj: dict) -> Vocab:
    v = Vocab(tokens=[], min_freq=0, reversed_tokens=[])
    v.idx_to_token = list(obj["idx_to_token"])
    v.token_to_idx = {t: i for i, t in enumerate(v.idx_to_token)}
    return v


def _torch_load_safe(path: str, device: torch.device, weights_only: bool = True):
    # weights_only=True is supported in newer PyTorch; keep fallback for compatibility.
    try:
        return torch.load(path, map_location=device, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint_any(path: str, device: torch.device, weights_only: bool = True) -> Dict[str, Any]:
    obj = _torch_load_safe(path, device=device, weights_only=weights_only)

    # Case 1: raw state_dict
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return {"model_state_dict": obj}

    # Case 2: full checkpoint dict
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj

    raise ValueError(
        f"Unrecognized checkpoint format: type={type(obj)} "
        f"keys={list(obj.keys()) if isinstance(obj, dict) else 'N/A'}"
    )


def _build_vocab_from_data(cfg: dict) -> Tuple[Vocab, Vocab]:
    dcfg = cfg["data"]
    train_path = dcfg["train_path"]
    max_train_examples = dcfg.get("max_train_examples")
    src_tokens, tgt_tokens = tokenize(train_path, max_train_examples)
    src_vocab = Vocab(src_tokens, min_freq=0, reversed_tokens=["<pad>", "<bos>", "<eos>"])
    tgt_vocab = Vocab(tgt_tokens, min_freq=0, reversed_tokens=["<pad>", "<bos>", "<eos>"])
    return src_vocab, tgt_vocab


def build_model(cfg: dict, src_vocab: Vocab, tgt_vocab: Vocab) -> EncoderDecoder:
    mcfg = cfg["model"]
    typ = str(mcfg["type"]).lower()

    if typ == "transformer":
        p = mcfg["transformer"]
        enc = TransformerEncoder(
            vocab_size=len(src_vocab),
            key_size=p["num_hiddens"],
            query_size=p["num_hiddens"],
            value_size=p["num_hiddens"],
            num_hiddens=p["num_hiddens"],
            norm_shape=[p["num_hiddens"]],
            ffn_num_input=p["num_hiddens"],
            ffn_num_hiddens=p["ffn_num_hiddens"],
            num_heads=p["num_heads"],
            num_layers=p["num_layers"],
            dropout=p["dropout"],
            use_bias=p.get("use_bias", False),
        )
        dec = TransformerDecoder(
            vocab_size=len(tgt_vocab),
            key_size=p["num_hiddens"],
            query_size=p["num_hiddens"],
            value_size=p["num_hiddens"],
            num_hiddens=p["num_hiddens"],
            norm_shape=[p["num_hiddens"]],
            ffn_num_input=p["num_hiddens"],
            ffn_num_hiddens=p["ffn_num_hiddens"],
            num_heads=p["num_heads"],
            num_layers=p["num_layers"],
            dropout=p["dropout"],
            use_bias=p.get("use_bias", False),
        )
        return EncoderDecoder(enc, dec)

    if typ == "seq2seq":
        p = mcfg["seq2seq"]
        enc = Seq2seqEncoder(
            vocab_size=len(src_vocab),
            embedding_size=p["embed_size"],
            num_hiddens=p["num_hiddens"],
            num_layers=p["num_layers"],
            dropout=p.get("dropout", 0.0),
            bidirectional=p.get("bidirectional", True),
        )

        # Repo variants may differ slightly in constructor signature. Try named args first.
        decoder_exceptions = []
        for kwargs in [
            dict(
                vocab_size=len(tgt_vocab),
                embedding_size=p["embed_size"],
                num_hiddens=p["num_hiddens"],
                num_layers=p["num_layers"],
                num_heads=p.get("num_heads", 8),
                dropout=p.get("dropout", 0.0),
                bidirectional=p.get("bidirectional", True),
            ),
            dict(
                vocab_size=len(tgt_vocab),
                embedding_size=p["embed_size"],
                num_hiddens=p["num_hiddens"],
                num_layers=p["num_layers"],
                num_heads=p.get("num_heads", 8),
                dropout=p.get("dropout", 0.0),
            ),
            dict(
                vocab_size=len(tgt_vocab),
                embedding_size=p["embed_size"],
                num_hiddens=p["num_hiddens"],
                num_layers=p["num_layers"],
                dropout=p.get("dropout", 0.0),
            ),
        ]:
            try:
                dec = Seq2seqAttentionDecoder(**kwargs)
                return EncoderDecoder(enc, dec)
            except TypeError as e:
                decoder_exceptions.append(str(e))
                continue

        raise TypeError(
            "Failed to construct Seq2seqAttentionDecoder. Tried multiple signatures. "
            + " | ".join(decoder_exceptions)
        )

    raise ValueError(f"Unsupported model.type={typ!r}")


def load_all(cfg_path: str, device: torch.device):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _seed_everything(int(cfg.get("seed", 42)))
    ckcfg = cfg.get("checkpoint", {})
    resume_from = ckcfg.get("resume_from")
    init_from = ckcfg.get("init_from")
    weights_only = bool(ckcfg.get("weights_only", True))

    ckpt = None
    st = LoadedState()

    ckpt_path = resume_from or init_from
    if ckpt_path:
        ckpt = load_checkpoint_any(ckpt_path, device=device, weights_only=weights_only)

    if ckpt is not None and "src_vocab" in ckpt and "tgt_vocab" in ckpt:
        src_vocab = _vocab_from_json(ckpt["src_vocab"])
        tgt_vocab = _vocab_from_json(ckpt["tgt_vocab"])
    else:
        src_vocab, tgt_vocab = _build_vocab_from_data(cfg)

    net = build_model(cfg, src_vocab, tgt_vocab).to(device)

    if ckpt is not None:
        net.load_state_dict(ckpt["model_state_dict"])
        st.epoch = int(ckpt.get("epoch", 0))
        st.step = int(ckpt.get("step", 0))
        st.best_bleu = float(ckpt.get("best_bleu", 0.0))
        st.optimizer_state = ckpt.get("optimizer_state_dict")
        st.scheduler_state = ckpt.get("scheduler_state_dict")


    return net, src_vocab, tgt_vocab, st, cfg


