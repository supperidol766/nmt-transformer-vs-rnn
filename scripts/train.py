from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from typing import Any, Dict, Iterable, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from nmt.common.device import try_GPU
from nmt.layers.attention import sequence_mask
from nmt.data.dataset import tokenize, build_array_nmt, TextDataset2

from scripts.entrypoint import load_all, _vocab_to_json


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred: torch.Tensor, label: torch.Tensor, valid_lens: torch.Tensor) -> torch.Tensor:
        # pred: (B, T, V), label: (B, T), valid_lens: (B,)
        weights = torch.ones_like(label, dtype=torch.float32)
        weights = sequence_mask(weights, valid_lens)
        self.reduction = "none"
        unweighted = super().forward(pred.permute(0, 2, 1), label)
        weighted = unweighted * weights
        return weighted.sum() / weights.sum().clamp_min(1.0)


def grad_clipping(net: nn.Module, max_norm: float) -> None:
    params = [p for p in net.parameters() if p.requires_grad]
    if params:
        nn.utils.clip_grad_norm_(params, max_norm=max_norm)


def _dataloader_kwargs(cfg_data: dict, shuffle: bool) -> dict:
    num_workers = int(cfg_data.get("num_workers", 0))
    pin_memory = bool(cfg_data.get("pin_memory", False))
    kwargs = dict(
        batch_size=int(cfg_data["batch_size"]),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # prefetch_factor only valid when num_workers > 0
    if num_workers > 0 and cfg_data.get("prefetch_factor") is not None:
        kwargs["prefetch_factor"] = int(cfg_data["prefetch_factor"])
    return kwargs


def make_dataloaders(cfg: dict, src_vocab, tgt_vocab):
    dcfg = cfg["data"]
    num_steps = int(dcfg["num_steps"])

    # train
    s_train, t_train = tokenize(dcfg["train_path"], dcfg.get("max_train_examples"))
    s_arr, s_len = build_array_nmt(s_train, src_vocab, num_steps)
    t_arr, t_len = build_array_nmt(t_train, tgt_vocab, num_steps)
    train_ds = TextDataset2(s_arr, s_len, t_arr, t_len)

    # dev
    s_dev, t_dev = tokenize(dcfg["dev_path"], dcfg.get("max_dev_examples"))
    s_arr_d, s_len_d = build_array_nmt(s_dev, src_vocab, num_steps)
    t_arr_d, t_len_d = build_array_nmt(t_dev, tgt_vocab, num_steps)
    dev_ds = TextDataset2(s_arr_d, s_len_d, t_arr_d, t_len_d)

    train_loader = DataLoader(train_ds, **_dataloader_kwargs(dcfg, shuffle=True))
    dev_loader = DataLoader(dev_ds, **_dataloader_kwargs(dcfg, shuffle=False))
    return train_loader, dev_loader


def _get_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, total_steps: int):
    tcfg = cfg["train"]
    warmup_steps = int(tcfg.get("warmup_steps", 4000))
    warmup_steps = max(1, warmup_steps)

    def lr_lambda(step_idx: int) -> float:
        step = max(1, step_idx + 1)
        if step <= warmup_steps:
            return step / warmup_steps
        # cosine decay to 10% (avoid hitting exact 0 too early)
        remain = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / remain)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def greedy_decode_batch(net, src: torch.Tensor, src_valid_len: torch.Tensor, bos_id: int, eos_id: int, max_len: int):
    device = src.device
    net.eval()

    enc_out = net.encoder(src, src_valid_len)
    dec_state = net.decoder.init_state(enc_out, src_valid_len)

    batch_size = src.size(0)
    dec_x = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    outs: List[torch.Tensor] = []

    for _ in range(max_len):
        y, dec_state = net.decoder(dec_x[:, -1:], dec_state)  # (B,1,V)
        next_tok = y[:, -1, :].argmax(dim=-1, keepdim=True)   # (B,1)
        outs.append(next_tok)
        dec_x = torch.cat([dec_x, next_tok], dim=1)

    return torch.cat(outs, dim=1) if outs else torch.empty((batch_size, 0), dtype=torch.long, device=device)


def _trim_ids(ids: Iterable[int], eos_id: int, pad_id: int) -> List[int]:
    out: List[int] = []
    for x in ids:
        xi = int(x)
        if xi == eos_id:
            break
        if xi != pad_id:
            out.append(xi)
    return out


def _bleu_sentence(pred_tokens: List[str], label_tokens: List[str], k: int = 4, smooth: float = 1.0) -> float:
    # Light-weight sentence BLEU with smoothing.
    if len(pred_tokens) == 0:
        return 0.0
    pred_len, label_len = len(pred_tokens), len(label_tokens)
    bp = math.exp(min(0.0, 1.0 - label_len / max(pred_len, 1)))
    score = bp
    for n in range(1, k + 1):
        if pred_len < n:
            break
        pred_ngrams = {}
        label_ngrams = {}
        for i in range(pred_len - n + 1):
            g = tuple(pred_tokens[i:i + n])
            pred_ngrams[g] = pred_ngrams.get(g, 0) + 1
        for i in range(max(0, label_len - n + 1)):
            g = tuple(label_tokens[i:i + n])
            label_ngrams[g] = label_ngrams.get(g, 0) + 1
        match = 0
        total = max(1, pred_len - n + 1)
        for g, c in pred_ngrams.items():
            match += min(c, label_ngrams.get(g, 0))
        precision = (match + smooth) / (total + smooth)
        score *= precision ** (0.5 ** n)
    return float(score)


@torch.no_grad()
def evaluate_bleu_quick(net, dev_loader, tgt_vocab, max_len: int, max_batches: int = 50) -> float:
    net.eval()
    bos_id = tgt_vocab["<bos>"]
    eos_id = tgt_vocab["<eos>"]
    pad_id = tgt_vocab["<pad>"]

    scores: List[float] = []
    for bi, batch in enumerate(dev_loader):
        if bi >= max_batches:
            break
        X, X_valid_len, Y, Y_valid_len = batch
        device = next(net.parameters()).device
        X = X.to(device)
        X_valid_len = X_valid_len.to(device)
        Y = Y.to(device)
        Y_valid_len = Y_valid_len.to(device)

        pred_ids = greedy_decode_batch(net, X, X_valid_len, bos_id, eos_id, max_len)
        for i in range(pred_ids.size(0)):
            pred_seq = _trim_ids(pred_ids[i].tolist(), eos_id, pad_id)

            gold_row = Y[i].tolist()
            if gold_row and int(gold_row[0]) == bos_id:
                gold_row = gold_row[1:]
            gold_seq = _trim_ids(gold_row, eos_id, pad_id)

            if hasattr(tgt_vocab, "to_tokens"):
                pred_tokens = tgt_vocab.to_tokens(pred_seq)
                gold_tokens = tgt_vocab.to_tokens(gold_seq)
            else:
                pred_tokens = [tgt_vocab.idx_to_token[j] for j in pred_seq]
                gold_tokens = [tgt_vocab.idx_to_token[j] for j in gold_seq]
            scores.append(_bleu_sentence(pred_tokens, gold_tokens))

    return float(sum(scores) / max(1, len(scores)))


def _ensure_csv(path: str, header: List[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def _append_csv(path: str, row: List[Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def _save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def _prune_epoch_ckpts(ckpt_dir: str, keep_last_k: int) -> None:
    if keep_last_k <= 0 or not os.path.isdir(ckpt_dir):
        return
    files = []
    for fn in os.listdir(ckpt_dir):
        if fn.startswith("epoch_") and fn.endswith(".pt"):
            try:
                ep = int(fn[len("epoch_"):-3])
            except ValueError:
                continue
            files.append((ep, os.path.join(ckpt_dir, fn)))
    files.sort()
    for _, path in files[:-keep_last_k]:
        try:
            os.remove(path)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
    args = ap.parse_args()

    device = try_GPU() if args.device == "auto" else torch.device(args.device)

    net, src_vocab, tgt_vocab, st, cfg = load_all(args.config, device=device)
    tcfg = cfg["train"]
    dcfg = cfg["data"]

    train_loader, dev_loader = make_dataloaders(cfg, src_vocab, tgt_vocab)

    save_dir = tcfg["save_dir"]
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # persist config snapshot
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(save_dir, "config.yaml"))

    metrics_csv = os.path.join(save_dir, "metrics.csv")
    lr_csv = os.path.join(save_dir, "lr.csv")
    _ensure_csv(metrics_csv, ["epoch", "train_loss", "dev_bleu"])
    _ensure_csv(lr_csv, ["global_step", "lr"])

    criterion = MaskedSoftmaxCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=float(tcfg["lr"]))

    total_steps = int(tcfg["epochs"]) * max(1, len(train_loader))
    scheduler = _get_lr_scheduler(optimizer, cfg, total_steps=total_steps)

    if st.optimizer_state is not None:
        optimizer.load_state_dict(st.optimizer_state)
    if st.scheduler_state is not None:
        scheduler.load_state_dict(st.scheduler_state)

    start_epoch = int(st.epoch)
    global_step = int(st.step)
    best_bleu = float(st.best_bleu)

    bos_id = tgt_vocab["<bos>"]
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    save_every_epochs = int(tcfg.get("save_every_epochs", 1))
    keep_last_k = int(tcfg.get("keep_last_k", 3))
    eval_max_batches = int(tcfg.get("eval_max_batches", 50))
    eval_num_steps = int(tcfg.get("eval_num_steps", dcfg["num_steps"]))

    print(f"[INFO] device={device} model={cfg['model']['type']} start_epoch={start_epoch} step={global_step}")

    for epoch in range(start_epoch, int(tcfg["epochs"])):
        net.train()
        running_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]

            # Teacher forcing input.
            # Robust to either label format:
            # - if Y already begins with <bos>, use Y[:, :-1]
            # - otherwise prepend <bos>
            if Y.size(1) > 0 and bool(torch.all(Y[:, 0] == bos_id)):
                dec_input = Y[:, :-1]
            else:
                bos = torch.full((Y.size(0), 1), bos_id, dtype=torch.long, device=device)
                dec_input = torch.cat([bos, Y[:, :-1]], dim=1)

            optimizer.zero_grad(set_to_none=True)
            Y_hat, _ = net(X, dec_input, X_valid_len)
            loss = criterion(Y_hat, Y, Y_valid_len)
            loss.backward()

            grad_clipping(net, grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += float(loss.item())
            num_batches += 1

            _append_csv(lr_csv, [global_step, optimizer.param_groups[0]["lr"]])

        train_loss = running_loss / max(1, num_batches)
        dev_bleu = evaluate_bleu_quick(
            net,
            dev_loader,
            tgt_vocab,
            max_len=eval_num_steps,
            max_batches=eval_max_batches,
        )
        _append_csv(metrics_csv, [epoch + 1, train_loss, dev_bleu])

        improved = dev_bleu >= best_bleu
        if improved:
            best_bleu = dev_bleu

        payload = {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch + 1,   # next epoch index for resume
            "step": global_step,
            "best_bleu": best_bleu,
            "src_vocab": _vocab_to_json(src_vocab),
            "tgt_vocab": _vocab_to_json(tgt_vocab),
            "config": cfg,
        }

        # always save last
        _save_checkpoint(os.path.join(ckpt_dir, "last.pt"), payload)

        if (epoch + 1) % save_every_epochs == 0:
            _save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt"), payload)
            _prune_epoch_ckpts(ckpt_dir, keep_last_k)

        if improved:
            _save_checkpoint(os.path.join(ckpt_dir, "best.pt"), payload)

        print(
            f"[Epoch {epoch+1}/{tcfg['epochs']}] "
            f"train_loss={train_loss:.4f} dev_bleu={dev_bleu:.4f} best_bleu={best_bleu:.4f}"
        )


if __name__ == "__main__":
    main()