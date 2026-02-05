import torch
import math
from torch import nn
from ..layers.attention import MultiHeadAttention


# FFN
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, dropout=0.1, bias=True, **kwargs):
        super().__init__()

        self.gate_up = nn.Linear(ffn_num_input, 2*ffn_num_hiddens, bias=bias)
        self.down = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        self.SiLU = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(self.dropout(self.SiLU(gate) * up))


# Add and norm
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropouts, **kwargs):
        super().__init__()
        self.dropouts = nn.Dropout(dropouts)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.ln(self.dropouts(y) + x)


# Encoder
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, x, *args):
        raise NotImplementedError

class PositionEncode(nn.Module):
    def __init__(self, dropout, hidden_size, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        p = torch.zeros(1, max_len, hidden_size)
        x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, hidden_size, 2, dtype=torch.float32) / hidden_size)
        p[:, :, 0::2] = torch.sin(x)
        p[:, :, 1::2] = torch.cos(x)

        self.register_buffer("p", p)

    def forward(self, x, start_pos: int = 0):
        # x: (batch, steps, hidden)
        end_pos = start_pos + x.shape[1]
        return self.dropout(x + self.p[:, start_pos:end_pos, :])



class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super().__init__()
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, dropout, num_heads, bias=use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens, dropout=dropout)

    def forward(self, x, valid_lens):
        y = self.attention(x, x, x, valid_lens)
        x = self.addnorm1(x, y)
        y = self.ffn(x)
        return self.addnorm2(x, y)

class TransformerEncoder(Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, dropout, num_layers, use_bias=False, **kwargs):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.dropout = nn.Dropout(dropout)
        self.embbedding = nn.Embedding(vocab_size, num_hiddens)
        self.net = nn.Sequential()
        self.position = PositionEncode(dropout, num_hiddens)
        for i in range(num_layers):
            self.net.add_module("EncoderBlock{}".format(i), EncoderBlock(key_size, query_size, value_size, num_hiddens,norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, x, valid_lens):
        x = self.embbedding(x)
        x = self.position(x*math.sqrt(self.num_hiddens))
        for net in self.net:
            x = net(x, valid_lens)
        return x


# Decoder
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, x, state):
        raise NotImplementedError

class AttentionDecoder(Decoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, dropout, i, use_bias=False, **kwargs):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.dropout = nn.Dropout(dropout)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, dropout, num_heads)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, dropout, num_heads)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.addnorm3 = AddNorm(norm_shape, dropout)
        self.FFN = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens, use_bias=use_bias)

    def forward(self, x, state):
        enc_outputs, valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_value = x
        else:
            key_value = torch.cat((state[2][self.i], x), axis=1)
        state[2][self.i] = key_value
        if self.training:
            batch_size, num_steps, _ = x.shape
            dec_valid_lens = torch.arange(1, num_steps+1, device=x.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        x2 = self.attention1(x, key_value, key_value, dec_valid_lens)
        y = self.addnorm1(x, x2)
        y2 = self.attention2(y, enc_outputs, enc_outputs, valid_lens)
        z = self.addnorm2(y, y2)
        y3 = self.FFN(z)
        z2 = self.addnorm3(z, y3) #torch.Size([batch-size, 20, 512])
        return z2, state



class TransformerDecoder(AttentionDecoder):
    def __init__(self,vocab_size ,key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, dropout, num_layers, use_bias=False, **kwargs):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(num_layers):
            self.net.add_module('block'+f'{i}', DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, dropout, i, use_bias))

        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos = PositionEncode(dropout=dropout, hidden_size=num_hiddens)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None]*self.num_layers]

    def forward(self, x, state):
        start_pos = 0
        if (not self.training) and (state[2][0] is not None):
            start_pos = state[2][0].shape[1]

        x = self.embedding(x) * math.sqrt(self.num_hiddens)
        x = self.pos(x, start_pos=start_pos)

        for layer in self.net:
            x, state = layer(x, state)
        return self.dense(x), state