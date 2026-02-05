import torch
import math
import torch.nn as nn


def sequence_mask(x, valid_lens, value=0):
    maxlen = x.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device = x.device)[None, :] < valid_lens[:, None]
    x[~mask] = value
    return x

def masked_softmax(x, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(x, dim=-1)
    else:
        shape = x.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        x = sequence_mask(x.reshape(-1, shape[-1]), valid_lens, value=-1e9)
        return nn.functional.softmax(x.reshape(shape), dim=-1)

class Dot_attention(nn.Module):
    def __init__(self, q_size, k_size, v_size, hidden_size, dropout):
        super(Dot_attention, self).__init__()
        self.W_q = nn.Linear(q_size, hidden_size, bias=False)
        self.W_k = nn.Linear(k_size, hidden_size, bias=False)
        self.W_v = nn.Linear(v_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, valid_lens=None, num_heads=1):
        q = self.transpose_in(self.W_q(q), num_heads)
        k = self.transpose_in(self.W_k(k), num_heads)
        v = self.transpose_in(self.W_v(v), num_heads)

        d = q.shape[-1]
        attention_score = masked_softmax(torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(d), valid_lens)
        return torch.bmm(self.dropout(attention_score), v), attention_score

    def project_hod_kv(self, k, v, num_heads=1):
        k_proj = self.transpose_in(self.W_k(k), num_heads)
        v_proj = self.transpose_in(self.W_v(v), num_heads)
        return k_proj, v_proj

    def forward_with_hod(self, q, k_proj, v_proj, valid_lens=None, num_heads=1):
        q = self.transpose_in(self.W_q(q), num_heads)
        d = q.shape[-1]
        attn = masked_softmax(torch.bmm(q, k_proj.transpose(-2, -1)) / math.sqrt(d), valid_lens)
        out = torch.bmm(self.dropout(attn), v_proj)
        return out, attn

    def transpose_in(self, x, num_heads): # x(batch, q-k, num_hidden)
        x = x.reshape(x.shape[0], x.shape[1], num_heads, -1).permute(0, 2, 1, 3) # x(batch, num_heads, q-k, num_hidden/num_heads)
        return x.reshape(-1, x.shape[2], x.shape[3]) # x(batch*num_heads, q-k, num_hidden/num_heads)

class MultiHeadAttention(nn.Module):
    def __init__(self, q_size, k_size, v_size, hidden_size, dropout, num_heads:int, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.attention = Dot_attention(q_size, k_size, v_size, hidden_size, dropout)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.num_heads = num_heads

    def forward(self, q, k, v, valid_lens=None, need_weights=False):
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        output, self.attention_weights = self.attention(q, k, v, num_heads=self.num_heads,
                                                        valid_lens=valid_lens)
        out_contact = self.transpose_out(output, num_heads=self.num_heads)
        return (self.W_o(out_contact), self.attention_weights) if need_weights else self.W_o(out_contact)

    def precompute_kv(self, k, v, valid_lens=None):
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        k_proj, v_proj = self.attention.project_hod_kv(k, v, num_heads=self.num_heads)
        return k_proj, v_proj, valid_lens

    def forward_with_kv(self, q, k_proj, v_proj, valid_lens=None, need_weights=False): # Reduce the usage of memory in seq2seq
        output, self.attention_weights = self.attention.forward_with_hod(
            q, k_proj, v_proj, valid_lens=valid_lens, num_heads=self.num_heads
        )
        out_contact = self.transpose_out(output, num_heads=self.num_heads)
        return (self.W_o(out_contact), self.attention_weights) if need_weights else self.W_o(out_contact)

    def transpose_out(self, x, num_heads): # x(batch*num_heads, q-k, num_hidden/num_heads)
        x = x.reshape(-1, num_heads, x.shape[1], x.shape[2]).permute(0, 2, 1, 3) # x(batch, q-k, num_heads, num_hidden/num_heads)
        return x.reshape(x.shape[0], x.shape[1], -1) # x(batch, q-k, num_hidden)