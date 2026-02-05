import math
import torch
import matplotlib.pyplot as plt
from GitHub.nmt.common.device import try_GPU
from GitHub.nmt.data.dataset import truncate_pad
from infer import preprocess_sentence


def _plot_heatmap(attn_2d, src_tokens, tgt_tokens, title="", save_path=None, dpi=160):
    attn = attn_2d.detach().cpu()

    fig_w = max(6, 0.40 * len(src_tokens))
    fig_h = max(4, 0.35 * len(tgt_tokens))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    im = ax.imshow(attn, aspect="auto")
    ax.set_title(title)

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=60, ha="right")
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens)

    ax.set_xlabel("source")
    ax.set_ylabel("target")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def _plot_heads_grid(attn_heads, src_tokens, tgt_tokens, title="", save_path=None, cols=4, dpi=160):
    """
    attn_heads: (H, tgt_len, src_len) torch.Tensor
    """
    A = attn_heads.detach().cpu()
    H = A.shape[0]
    rows = math.ceil(H / cols)

    fig_w = max(10, 0.35 * len(src_tokens) * cols / 2)
    fig_h = max(6, 0.30 * len(tgt_tokens) * rows / 2)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)
    axes = axes.reshape(rows, cols)

    for h in range(rows * cols):
        r, c = divmod(h, cols)
        ax = axes[r, c]
        ax.axis("off")
        if h >= H:
            continue
        ax.axis("on")
        im = ax.imshow(A[h], aspect="auto")
        ax.set_title(f"head {h}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def _reshape_attn_BHQS_to_BH_S(attn, num_heads):
    BH, Q, S = attn.shape
    assert BH % num_heads == 0, f"BH={BH} not divisible by H={num_heads}"
    B = BH // num_heads
    return attn.view(B, num_heads, Q, S)

@torch.no_grad()
def visualize_seq2seq_attention(
    net,
    src_sentence: str,
    src_vocab,
    tgt_vocab,
    num_steps: int,
    device=None,
    save_prefix="seq2seq_attn",
    plot_heads=True,
):
    if device is None:
        device = try_GPU()

    net.eval()
    net.to(device)
    src_words = preprocess_sentence(src_sentence)
    src_ids = src_vocab[src_words] + [src_vocab["<eos>"]]
    src_len_true = len(src_ids)

    enc_valid_len = torch.tensor([src_len_true], device=device)
    src_ids_pad = truncate_pad(src_ids, num_steps, src_vocab["<pad>"])
    enc_x = torch.tensor(src_ids_pad, dtype=torch.long, device=device).unsqueeze(0)  # (1,S)

    enc_outputs = net.encoder(enc_x, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    bos = tgt_vocab["<bos>"]
    eos = tgt_vocab["<eos>"]

    dec_x = torch.tensor([[bos]], dtype=torch.long, device=device)  # (1,1)

    attn_steps = []
    pred_token_ids = []

    num_heads = net.decoder.attention.num_heads

    for _ in range(num_steps):
        y, dec_state = net.decoder(dec_x, dec_state)      # y: (B,1,V)
        dec_x = y.argmax(dim=2)                           # (B,1)
        pred_id = int(dec_x.item())
        w = net.decoder.attention.attention_weights # torch.Tensor
        # reshape -> (B,H,1,S) -> 取B=0,Q=0 => (H,S)
        w_bhqs = _reshape_attn_BHQS_to_BH_S(w, num_heads)
        w_hs = w_bhqs[0, :, 0, :] # (H,S)
        w_hs = w_hs[:, :src_len_true] # (H,src_len_true)
        attn_steps.append(w_hs)

        if pred_id == eos:
            pred_token_ids.append(eos)
            break
        pred_token_ids.append(pred_id)

    # (tgt_len, H, src_len) -> (H, tgt_len, src_len)
    A_heads = torch.stack(attn_steps, dim=0).permute(1, 0, 2).contiguous()
    A_avg = A_heads.mean(dim=0)  # (tgt_len, src_len)

    src_tokens = src_words + ["<eos>"]
    tgt_tokens = tgt_vocab.to_tokens(pred_token_ids)

    _plot_heatmap( # plot average
        A_avg,
        src_tokens,
        tgt_tokens,
        title="Seq2seq cross-attn (avg over heads)",
        save_path=f"{save_prefix}_avg.png",
    )

    if plot_heads: # plot 8 heads graph
        _plot_heads_grid(
            A_heads,
            src_tokens,
            tgt_tokens,
            title="Seq2seq cross-attn (per head)",
            save_path=f"{save_prefix}_heads.png",
            cols=4,
        )

    pred_text = " ".join(tgt_tokens)
    return pred_text, A_avg, A_heads

@torch.no_grad()
def visualize_transformer_attention(
    net,
    src_sentence: str,
    src_vocab,
    tgt_vocab,
    num_steps: int,
    device=None,
    save_prefix="transformer_attn",
    layer_index=-1, # last layer
    plot_heads=True,
):

    if device is None:
        device = try_GPU()

    net.eval()
    net.to(device)

    src_words = preprocess_sentence(src_sentence)
    src_ids = src_vocab[src_words] + [src_vocab["<eos>"]]
    src_len_true = len(src_ids)

    enc_valid_len = torch.tensor([src_len_true], device=device)
    src_ids_pad = truncate_pad(src_ids, num_steps, src_vocab["<pad>"])
    enc_x = torch.tensor(src_ids_pad, dtype=torch.long, device=device).unsqueeze(0)  # (1,S)

    enc_outputs = net.encoder(enc_x, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)

    blocks = list(net.decoder.net) # nn.Sequential 可迭代
    L = len(blocks)
    if layer_index < 0:
        layer_index = L + layer_index
    assert 0 <= layer_index < L, f"layer_index out of range: got {layer_index}, but L={L}"
    block = blocks[layer_index]

    bos = tgt_vocab["<bos>"]
    eos = tgt_vocab["<eos>"]

    dec_x = torch.tensor([[bos]], dtype=torch.long, device=device)  # (1,1)

    attn_steps = []
    pred_token_ids = []

    num_heads = block.attention2.num_heads

    for _ in range(num_steps):
        y, dec_state = net.decoder(dec_x, dec_state) # y: (B,1,V)
        dec_x = y.argmax(dim=2) # (B,1)
        pred_id = int(dec_x.item())

        w = block.attention2.attention_weights
        w_bhqs = _reshape_attn_BHQS_to_BH_S(w, num_heads)
        w_hs = w_bhqs[0, :, 0, :] # (H,S)
        w_hs = w_hs[:, :src_len_true] # (H,src_len_true)

        attn_steps.append(w_hs)

        if pred_id == eos:
            pred_token_ids.append(eos)
            break
        pred_token_ids.append(pred_id)

    A_heads = torch.stack(attn_steps, dim=0).permute(1, 0, 2).contiguous()
    A_avg = A_heads.mean(dim=0)  # (tgt_len, src_len)

    src_tokens = src_words + ["<eos>"]
    tgt_tokens = tgt_vocab.to_tokens(pred_token_ids)

    _plot_heatmap( # plot average
        A_avg,
        src_tokens,
        tgt_tokens,
        title=f"Transformer cross-attn (layer {layer_index}, avg over heads)",
        save_path=f"{save_prefix}_layer{layer_index}_avg.png",
    )

    if plot_heads: # plot 8 heads graph
        _plot_heads_grid(
            A_heads,
            src_tokens,
            tgt_tokens,
            title=f"Transformer cross-attn (layer {layer_index}, per head)",
            save_path=f"{save_prefix}_layer{layer_index}_heads.png",
            cols=4,
        )

    pred_text = " ".join(tgt_tokens)
    return pred_text, A_avg, A_heads