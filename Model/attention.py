import torch
import torch.nn as nn
import math
from Auxiliary.utils import clones


class BahdanauAttention(nn.Module):
    '''
    Attention similar to
    Neural Machine Translation by Jointly Learning to Align and Translate
    https://arxiv.org/pdf/1409.0473.pdf
    '''
    def __init__(self, emb_dim, hidden_dim, bias=True, coverage=False):
        super(BahdanauAttention, self).__init__()
        proj_dim = 2 * emb_dim
        if coverage:
            proj_dim += 1
        self.alignment = nn.Linear(4 * hidden_dim, 2 * hidden_dim, bias=bias)
        self.v = nn.Linear(2 * hidden_dim, 1, bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.coverage = coverage

    def forward(self, h_j, s_prev, y_prev, coverage=None):
        '''
        :param h_j: encoder annotations/outputs h_1, ..., h_Tx (batch_size, max_enc_len, 2*hidden_dim) - (16, 400, 2*256)
        :param s_prev: previous decoder state s_i-1 e.g. (batch_size, 2*hidden_dim) - (16, 2*256)
        :param y_prev: decoder input symbols e.g. (batch_size, emb_dim) - (16, 300)
        :param coverage: coverage vector, e.g. (batch_size, enc_max_len) - (16, 400)
        :return: attention weights, attention vector
        '''

        enc_max_len = h_j.size(1)
        concat = torch.cat((s_prev.squeeze().unsqueeze(1).repeat(1, enc_max_len, 1), h_j), dim=-1)  # (16, 1*400, 2*256) and (16, 400, 2*256) -> (16, 400, 4*256)
        if self.coverage:
            concat = torch.cat((concat, coverage.unsqueeze(2)), dim=-1)  # (16, 400, 601)
        concat = self.alignment(concat)  # (16, 400, 2*256)
        concat = self.tanh(concat)  # (16, 400, 2*256)
        e_ij = self.v(concat)  # energies (16, 400, 1)
        a_tj = self.softmax(e_ij)  # (16, 400, 1)
        context_vector = torch.bmm(a_tj.permute(0, 2, 1), h_j).squeeze()  # (16, 2*256)

        att_vector = torch.cat((context_vector, y_prev), dim=-1)  # (16, 2*256+300) = (16, 812)

        new_coverage = None
        if self.coverage:
            new_coverage = coverage + a_tj.squeeze()

        return a_tj.squeeze(), att_vector, new_coverage  # (16, 400), (16, 812), (16, 400)


class LuongAttention(nn.Module):
    '''
        Attention similar to
        Effective Approaches to Attention-based Neural Machine Translation
        https://arxiv.org/abs/1508.04025
    '''
    def __init__(self, emb_dim, hidden_dim, bias=True, coverage=False):
        super(LuongAttention, self).__init__()
        #self.project_enc2emb = nn.Linear(2 * hidden_dim, emb_dim, bias=bias)
        #self.project_concat2emb = nn.Linear(2 * emb_dim, emb_dim, bias=bias)
        self.projector = nn.Linear(4 * hidden_dim, 2 * hidden_dim, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.coverage = coverage

    def forward(self, h_j, s_i, coverage=None):
        '''
        :param h_j: encoder annotations/outputs h_1, ..., h_Tx e.g.(batch_size, max_enc_len, 2 * hidden_dim) - (16, 400, 2*256)
        :param s_i: current decoder state e.g.(16, 1, 2*256)
        :param coverage: coverage vector, e.g.
        :return: attention weights, attention vector
        '''
        #h_j = self.project_enc2emb(h_j)  # (16, 400, 300)
        e_ij = torch.bmm(h_j, s_i.permute(0, 2, 1))  # (16, 400, 1)
        a_tj = self.softmax(e_ij)  # (16, 400, 1)
        context_vector = torch.bmm(a_tj.permute(0, 2, 1), h_j).squeeze()  # (16, 2*256)

        att_vector = torch.cat((context_vector, s_i.squeeze()), dim=-1)  # (16, 4*256)
        #att_vector = self.project_concat2emb(att_vector)
        att_vector = self.tanh(att_vector)  # (16, 4*256)
        att_vector = self.projector(att_vector)  # (16, 2*256)

        return a_tj.squeeze(), att_vector  # (16, 400), (16, 2*256)

class ScaledDotProductAttention(nn.Module):
    '''
        Scaled Dot Product Attention as in:
        https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
    '''
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask, dropout=None):
        d_k = q.size(-1)
        e_tj = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        e_tj = e_tj.masked_fill(mask == 0, -1e9)
        a_tj = self.softmax(e_tj)
        if dropout is not None:
            a_tj = dropout(a_tj)

        return torch.matmul(a_tj, v), a_tj # (4, 8, 400, 64)

class MultiHeadedAttention(nn.Module):
    '''
        Multi-Head Attention as in:
        https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
    '''
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attention = ScaledDotProductAttention()
        self.a_tj = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. # (4, 8, 400, 64)
        x, self.a_tj = self.attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)