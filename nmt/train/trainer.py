import os
import csv
import math
import torch
from torch import nn
from ..eval.bleu import evaluate_bleu
from ..common.device import try_GPU
from ..layers.attention import sequence_mask


class Accumulator:
    def __init__(self, n):
        self.data = [0.] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def grad_clipping(net, max_norm: float):
    if isinstance(net, torch.nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    nn.utils.clip_grad_norm_(params, max_norm=max_norm)

class Masked_Softmax_CEloss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_lens):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_lens)
        self.reduction = 'none'
        unweighted_loss = super(
            Masked_Softmax_CEloss, self
        ).forward(pred.permute(0, 2, 1), label)
        weighted_loss = unweighted_loss * weights
        sum_loss = weighted_loss.sum()
        num_tokens = weights.sum()
        loss = sum_loss / num_tokens
        return loss

def trainer(net, data_iter, lr, num_epochs, tgt_vocab, save_dir, data_test_iter, save, device=try_GPU(), load_from_saved=True):
    os.makedirs(save, exist_ok=True)
    metrics_path = os.path.join(save, "metrics.csv")
    lr_path = os.path.join(save, "lr.csv")
    step = 0

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "dev_bleu"])
        w.writeheader()

    with open(lr_path, "w", newline="", encoding="utf-8") as f:
        wlr = csv.DictWriter(f, fieldnames=["step", "lr"])
        wlr.writeheader()

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
             nn.init.xavier_uniform_(m.weight)

        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_uniform_(m._parameters[param])

        if isinstance(m, nn.GRUCell):
            for name, p in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(p)

    if load_from_saved:
        net.load_state_dict(torch.load(save_dir, map_location=device))
    else:
        net.apply(xavier_init_weights)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    total_steps = num_epochs * len(data_iter)
    warmup_steps = min(4000, int(0.1 * total_steps))

    def lr_lambda(step):
        step += 1
        if step <= warmup_steps:
            return step / warmup_steps
        # cosine decay to 0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    loss = Masked_Softmax_CEloss()
    net.train()

    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(2)
        for batch in data_iter:
            optimizer.zero_grad()
            x, x_valid_len, y, y_valid_len = [t.to(device) for t in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, y[:, :-1]], 1)#强制教学
            y_hat, _ = net(x, dec_input, x_valid_len)
            l = loss(y_hat, y, y_valid_len)
            l.sum().backward()

            for name, p in net.named_parameters():
                if p.requires_grad and p.grad is None:
                    print("NO_GRAD:", name, p.shape)

            grad_clipping(net, 1)
            num_tokens = y_valid_len.sum()
            optimizer.step()
            scheduler.step()
            step += 1
            with open(lr_path, "a", newline="", encoding="utf-8") as f:
                wlr = csv.DictWriter(f, fieldnames=["step", "lr"])
                wlr.writerow({"step": step, "lr": optimizer.param_groups[0]["lr"]})

            with torch.no_grad():
                metric.add(l.item(), 1)

        dev_bleu = evaluate_bleu(net, data_test_iter, tgt_vocab, num_steps=50, device=device, max_batches=50)
        train_loss = metric[0] / metric[1]
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "dev_bleu"])
            w.writerow({"epoch": epoch, "train_loss": train_loss, "dev_bleu": dev_bleu})

        print(f"epoch {epoch} | train_loss {train_loss:.4f} | dev_bleu {dev_bleu:.4f}")