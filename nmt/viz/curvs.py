import os
import pandas as pd
import matplotlib.pyplot as plt


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