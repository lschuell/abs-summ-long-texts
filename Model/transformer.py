import numpy as np
import torch
import torch.nn as nn
from Model.attention import MultiHeadedAttention
import math, copy
from constants import DATADUMP_PATH, CNNDM_DIR, START_DEC
from data import Vocabulary
from Auxiliary.utils import pickle_load, mask_oov, clones, pad_mask, future_and_pad_mask
import os
from torch.utils import data


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):  # (4, 400, 512)
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1) #  (max_len, 1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))  # (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # register buffer so that module preserved in state_dict
        self.register_buffer('pe', pe)  # (1, max_len, d_model)

    def forward(self, x):
        x += self.pe[:, :x.size(1)]

        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class DecoderTransformer(nn.Module):
    def __init__(self, embedding, N, vocab, device, dec_max_len, windower, pe, layer, d_model):
        super(DecoderTransformer, self).__init__()
        self.embedding = embedding
        self.pe = pe
        self.layers = clones(layer, N)
        self.norm = LayerNorm(d_model)
        self.d_model = d_model
        self.vocab = vocab
        self.projector_2out = nn.Linear(d_model, self.vocab.__len__())
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self._init_xavier()
        self.one_step_mode = False
        self.device = device
        self.windower = windower
        self.dec_max_len = dec_max_len
        self.sos = self.vocab[START_DEC]

    @classmethod
    def build(cls, d_model, d_ff, h, dropout):
        c = copy.deepcopy
        pe = PositionalEncoding(d_model, dropout)
        attention = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        dl = DecoderLayer(d_model, c(attention), c(attention), ff, dropout)

        return pe, dl, d_model

    def _init_xavier(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, m, tgt):
        bs = m.size(0)
        if self.training:
            return self._decode(src, m, tgt)

        else: # inference mode - greedy decoding
            outputs = []
            ys_prev = np.ones((bs, 1), dtype=int) * self.sos
            ys_prev = torch.from_numpy(ys_prev).to(self.device)
            for t in range(self.dec_max_len):
                dec_outputs, _ = self._decode(src, m, ys_prev)
                dec_outputs = dec_outputs[:, :, -1]
                outputs.append(dec_outputs)
                max_outputs = torch.argmax(dec_outputs, dim=1, keepdim=True)
                ys_prev = torch.cat((ys_prev, max_outputs), dim=1)

            return torch.stack(outputs).permute(1, 2, 0), None

    def _decode(self, src, m, tgt):
        src_mask = pad_mask(src, self.vocab)  # (batch_size, 1, enc_max_len)
        tgt_mask = future_and_pad_mask(tgt, self.vocab, self.device)
        out = self.embedding(tgt) * math.sqrt(self.embedding.weight.data.size(-1))
        out = self.pe(out)

        for layer in self.layers:
            out = layer(out, m, src_mask, tgt_mask)

        out = self.norm(out)
        out = self.projector_2out(out)  # (batch_size, max_dec_len, max_vocab)
        out = self.log_softmax(out)

        # (batch_size, max_vocab, max_dec_len)
        # TODO (optional): retrieve weights for visualizations
        return out.transpose(-2, -1), None

    def one_step_decode(self, src, m, ys_prev):
        dec_outputs, _ = self._decode(src, m, ys_prev)
        return dec_outputs[:, :, -1], None

class EncoderTransformer(nn.Module):
    def __init__(self, embedding, N, vocab, pe, layer, d_model):
        super(EncoderTransformer, self).__init__()
        self.embedding = embedding
        self.pe = pe
        self.layers = clones(layer, N)
        self.vocab = vocab
        self.norm = LayerNorm(d_model)
        self._init_xavier()

    @classmethod
    def build(cls, d_model, d_ff, h, dropout):
        pe = PositionalEncoding(d_model, dropout)
        attention = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        el = EncoderLayer(d_model, attention, ff, dropout)

        return pe, el, d_model

    def _init_xavier(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        src_mask = pad_mask(x, self.vocab)  # (batch_size, 1, max_enc_len)
        out = self.embedding(x) * math.sqrt(self.embedding.weight.data.size(-1))
        out = self.pe(out)
        for layer in self.layers:
            out = layer(out, src_mask)
        return self.norm(out)

    def hidden_final(self, x):
        # average memory representations to initialize a recurrent decoder
        return x.mean(dim=-2).unsqueeze(0)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)


    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, m, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

if __name__ == "__main__":

    device = torch.device("cpu")
    dec_max_len = 100
    data_val = f"data_CnnDm_val_50000_400_100_False.pickle"
    X_VAL = pickle_load(os.path.join(DATADUMP_PATH, data_val))
    val_loader = torch.utils.data.DataLoader(X_VAL, batch_size=4, num_workers=1,
                                             pin_memory=False, shuffle=True)

    VOCAB = Vocabulary(os.path.join(CNNDM_DIR, "vocab"), 50000)
    batch = None
    for i, b in enumerate(val_loader):
        batch = b
        if i == 0:
            break

    batch = batch.long()
    enc_inputs_oov = batch[:, :400]
    dec_inputs = batch[:, 400: 400 + 100]
    dec_targets_oov = batch[:, 400 + 100: 400 + 2 * 100]
    encoded_oovs = batch[:, 400 + 2 * 100:]
    enc_inputs_no_oov = mask_oov(enc_inputs_oov, VOCAB)

    d_model = 512; d_ff = 2048; h = 8; dropout = 0.1; N = 6
    embedding = nn.Embedding(50000, d_model)
    encoder = EncoderTransformer(embedding, N, VOCAB
                                 *EncoderTransformer.build(d_model, d_ff, h, dropout))
    decoder = DecoderTransformer(embedding,
                                 *DecoderTransformer.build(d_model, d_ff, h, dropout),
                                 N, VOCAB, device, dec_max_len)

    m = encoder(enc_inputs_no_oov)
    decoder
    out, _ = decoder(enc_inputs_no_oov, m, dec_inputs)

    print(out.size())


