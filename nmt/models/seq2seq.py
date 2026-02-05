import torch
from torch import nn
from GitHub.nmt.layers.attention import MultiHeadAttention


class Mogrifier(nn.Module):
    def __init__(self, x_size, h_size, **kwargs):
        super().__init__()
        self.Q = nn.ModuleList([
            nn.Linear(h_size, x_size) for _ in range(2)
        ])
        self.R = nn.ModuleList([
            nn.Linear(x_size, h_size) for _ in range(2)
        ])

    def forward(self, x, h):
        for i in range(4):
            if i % 2 == 0:
                i = i // 2
                x = x * (2 * torch.sigmoid(self.Q[i](h)))
            else:
                i = i // 2
                h = h * (2 * torch.sigmoid(self.R[i](x)))
        return x, h

class ResNormGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout=0.1, bidirectional=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dir = 2 if bidirectional else 1

        self.d_model = self.dir * hidden_size

        self.in_proj = nn.Identity() if input_size == self.d_model else nn.Linear(input_size, self.d_model, bias=False)

        self.norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(num_layers)])
        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)

        self.grus = nn.ModuleList([
            nn.GRU(input_size=self.d_model, hidden_size=hidden_size,
                   num_layers=1, bidirectional=bidirectional)
            for _ in range(num_layers)
        ])

    def forward(self, x, h=None):
        x = self.in_proj(x)  # (T,B,d_model)
        T, B, _ = x.shape

        if h is None:
            h = x.new_zeros(self.num_layers * self.dir, B, self.hidden_size)

        hs = []
        for i in range(self.num_layers):
            xin = self.norms[i](x)  # PreNorm

            hi = h[i*self.dir:(i+1)*self.dir].contiguous()  # (dir,B,H)
            y, hi = self.grus[i](xin, hi)                   # y: (T,B,d_model)

            x = x + self.alphas[i] * self.drop(y)           # ReZero
            hs.append(hi)

        hN = torch.cat(hs, dim=0)  # (L*dir,B,H)
        return x, hN

class Multi_hob_attention(nn.Module):
    def __init__(self, q_size, k_size, v_size, hidden_size, dropout, num_heads, num_hob=1, bias=False, **kwargs):
        super().__init__()
        self.attention_layer = \
            MultiHeadAttention(q_size, k_size, v_size, hidden_size, dropout, num_heads, bias=bias)
        self.refinement_GRU =  \
            nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.hobs = num_hob

    def forward(self, q, k, v, valid_lens=None, collect_hop_weights=False):
        self.hop_attention_weights = None
        k_p, v_p, valid_lens = self.attention_layer.precompute_kv(k, v, valid_lens)
        hop_weights = [] if collect_hop_weights else None
        for _ in range(self.hobs):
            content = self.attention_layer.forward_with_kv(q, k_p, v_p, valid_lens)
            if collect_hop_weights:
                hop_weights.append(self.attention_layer.attention_weights.detach().cpu())
            q2 = self.refinement_GRU(content.reshape(-1, q.size(-1)), q.reshape(-1, q.size(-1)))
            q = self.layer_norm(q + q2.unsqueeze(1))
        if collect_hop_weights:
            self.hop_attention_weights = hop_weights
        return self.attention_layer.forward_with_kv(q, k_p, v_p, valid_lens)

# Encoder
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, x, *args):
        raise NotImplementedError

class Seq2seqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0., bidirectional=True, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = ResNormGRULayer(embed_size, num_hiddens, num_layers, dropout, bidirectional=bidirectional)

    def forward(self, x, *args):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        output, state = self.rnn(x)
        return output, state

#Decoder
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

class Seq2seqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embedding_size, num_hiddens, num_layers, num_heads, dropout=0.1, enc_bi=True, **kwargs):
        super().__init__()
        enc_dim = 2 if enc_bi else 1
        self.attention = MultiHeadAttention(
            q_size=num_hiddens, k_size=enc_dim*num_hiddens, v_size=enc_dim*num_hiddens, hidden_size=num_hiddens, num_heads=num_heads, dropout=dropout
        )
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = ResNormGRULayer(embedding_size + num_hiddens, num_hiddens, num_layers, dropout=dropout, bidirectional=False)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self.enc = enc_bi

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs
        if self.enc:
            h_f = hidden_state[0::2]
            h_b = hidden_state[1::2]
            hidden_state = h_f + h_b
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, x, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        x = self.embedding(x).permute(1, 0, 2) # (S, B, H)
        outputs, self._attention_weights = [], []
        for i in x:
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens
            )
            i = torch.cat((context, torch.unsqueeze(i, dim=1)), dim=-1)
            out, hidden_state = self.rnn(i.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            if not self.training:
                self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights