import os
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def _trim(seq, eos_id, pad_id):
    # seq: 1D LongTensor
    out = []
    for x in seq.tolist():
        if x == eos_id:
            break
        if x != pad_id:
            out.append(x)
    return out

def _bleu_sentence(pred, label, k=4, smooth=True):
    pred_len, label_len = len(pred), len(label)
    if pred_len == 0:
        return 0.0
    k = min(k, pred_len, label_len)
    if k == 0:
        return 0.0

    bp = math.exp(min(0, 1 - label_len / pred_len))
    score = bp
    for n in range(1, k+1):
        pred_ngrams = Counter(tuple(pred[i:i+n]) for i in range(pred_len-n+1))
        label_ngrams = Counter(tuple(label[i:i+n]) for i in range(label_len-n+1))
        overlap = sum((pred_ngrams & label_ngrams).values())
        total = max(1, sum(pred_ngrams.values()))
        if smooth:
            score *= ((overlap + 1) / (total + 1)) ** (1.0 / k)
        else:
            score *= (overlap / total) ** (1.0 / k)
    return score

@torch.no_grad()
def greedy_decode_batch(net, x, x_valid_len, bos_id, eos_id, max_len):
    net.eval()
    enc_outputs = net.encoder(x, x_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, x_valid_len)

    B = x.shape[0]
    dec_x = torch.full((B, 1), bos_id, dtype=torch.long, device=x.device)
    preds = []
    for _ in range(max_len):
        y, dec_state = net.decoder(dec_x, dec_state)     # y: (B,1,V)
        dec_x = y.argmax(dim=2)                          # (B,1)
        preds.append(dec_x)
    return torch.cat(preds, dim=1)                       # (B, max_len)

@torch.no_grad()
def evaluate_bleu(net, data_iter, tgt_vocab, num_steps, device, max_batches=50):
    pad_id = tgt_vocab['<pad>']
    eos_id = tgt_vocab['<eos>']
    bos_id = tgt_vocab['<bos>']
    scores = []
    n_batches = 0
    for batch in data_iter:
        x, x_valid_len, y, y_valid_len = [t.to(device) for t in batch]
        pred = greedy_decode_batch(net, x, x_valid_len, bos_id, eos_id, max_len=num_steps)
        for i in range(x.shape[0]):
            pred_i = _trim(pred[i], eos_id, pad_id)
            lab_i  = _trim(y[i],   eos_id, pad_id)
            scores.append(_bleu_sentence(pred_i, lab_i, k=4))
        n_batches += 1
        if n_batches >= max_batches:
            break
    return float(sum(scores) / max(1, len(scores)))

def plot_lr(run_dir, warmup_steps=None):
    lr = pd.read_csv(os.path.join(run_dir, "lr.csv"))
    plt.figure()
    plt.plot(lr["step"], lr["lr"])
    if warmup_steps is not None:
        plt.axvline(warmup_steps)
    plt.xlabel("step"); plt.ylabel("lr"); plt.title("LR vs step")
    plt.savefig(os.path.join(run_dir, "lr_seq2seq.png"), dpi=200)
    plt.close()

def plot_loss_bleu(run_dir):
    m = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
    plt.figure()
    plt.plot(m["epoch"], m["train_loss"], label="train_loss")
    plt.plot(m["epoch"], m["dev_bleu"], label="dev_bleu")
    plt.legend()
    plt.xlabel("epoch")
    plt.title("Loss/BLEU vs epoch")
    plt.savefig(os.path.join(run_dir, "loss_bleu_seq2seq.png"), dpi=200)
    plt.close()