import os

import matplotlib.pyplot as plt
import pandas as pd


def _resolve_ste:contentReference[oaicite:8]{index=8}ep"
    if "global_step" in df.columns:
        return "global_step"
    raise KeyError(f"Neither 'step' nor 'global_step' found. Got columns={list(df.columns)}")


def plot_lr(run_dir, warmup_steps=None, out_name="lr_curve.png"):
    lr = pd.read_csv(os.path.join(run_dir, "lr.csv"))
    step_col = _resolve_step_col(lr)

    plt.figure()
    plt.plot(lr[step_col], lr["lr"])
    if warmup_steps is not None:
        plt.axvline(warmup_steps)
    plt.xlabel(step_col)
    plt.ylabel("lr")
    plt.title("LR vs step")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, out_name), dpi=200)
    plt.close()


def plot_loss_bleu(run_dir, out_name="loss_bleu_curve.png"):
    m = pd.read_csv(os.path.join(run_dir, "metrics.csv"))

    plt.figure()
    plt.plot(m["epoch"], m["train_loss"], label="train_loss")
    plt.plot(m["epoch"], m["dev_bleu"], label="dev_bleu")
    plt.legend()
    plt.xlabel("epoch")
    plt.title("Loss / BLEU vs epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, out_name), dpi=200)
    plt.close()


if __name__ == "__main__":
    pass
