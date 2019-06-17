import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import random
from Auxiliary.utils import mask_oov
from constants import START_DEC
from Model.windowing import EncoderSlider, StaticScheduler, DynamicScheduler, Windower
from Model.pointer import PointerNet
from Model.attention import LuongAttention, BahdanauAttention


# ------------------------------------------ENCODER-------------------------------------------------------------


class EncoderBiRNN(nn.Module):
    def __init__(self, Config, embedding, device):
        super(EncoderBiRNN, self).__init__()
        self.Config = Config
        self.embedding = embedding
        self.device = device

    def forward(self, x):
        raise NotImplementedError

    def hidden_final(self, enc_hidden):
        # concatenate forward and backward states
        _cat = lambda enc_hidden: torch.cat([enc_hidden[0:enc_hidden.size(0):2], enc_hidden[1:enc_hidden.size(0):2]], 2)

        if isinstance(enc_hidden, tuple):  # LSTM
            return tuple([_cat(h) for h in enc_hidden])
        else:  # GRU
            return _cat(enc_hidden)


class EncoderBiLSTM(EncoderBiRNN):
    def __init__(self, Config, embedding, device):
        super(EncoderBiLSTM, self).__init__(Config, embedding, device)
        self.bilstm = nn.LSTM(self.Config.emb_dim, self.Config.hidden_dim, self.Config.num_layers,
                              batch_first=True, bidirectional=True)

    def forward(self, x):
        h0 = torch.zeros(self.Config.num_layers * 2, x.size(0), self.Config.hidden_dim).to(self.device)
        c0 = torch.zeros(self.Config.num_layers * 2, x.size(0), self.Config.hidden_dim).to(self.device)

        x = self.embedding(x)
        out, enc_state = self.bilstm(x, (h0, c0))

        return out, enc_state

class EncoderBiGRU(EncoderBiRNN):
    def __init__(self, Config, embedding, device):
        super(EncoderBiGRU, self).__init__(Config, embedding, device)
        self.bigru = nn.GRU(self.Config.emb_dim, self.Config.hidden_dim, self.Config.num_layers,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        h0 = torch.zeros(self.Config.num_layers * 2, x.size(0), self.Config.hidden_dim).to(self.device)

        x = self.embedding(x)
        out, enc_state = self.bigru(x, h0)

        return out, enc_state

# ------------------------------------------DECODER-------------------------------------------------------------

class DecoderRNN(nn.Module):
    def __init__(self, embedding, Config, vocab, device, pointer, windower):
        super(DecoderRNN, self).__init__()
        self.Config = Config
        self.Config.rnn_type = self.Config.rnn_type.upper()
        self.Config.attention_type = self.Config.attention_type.title()
        self.embedding = embedding
        self.dec_max_len = None  # attribute is set on the fly according to batch max_dec_len
        self.tf_prob = self.Config.tf_prob
        self.vocab = vocab
        self.device = device
        self.windower = windower
        assert(self.Config.rnn_type in ['LSTM', 'GRU']), "Rnn-Type must be one of 'LSTM' or 'GRU' !"
        assert(self.Config.attention_type in ['Bahdanau', 'Luong']), "Attention-Type must be one of 'Bahdanau' or 'Luong' !"
        model_dim = self.Config.d_model \
            if self.Config.encoder.title() == 'Transformer' \
            else 2 * self.Config.hidden_dim
        emb_dim = self.Config.d_model \
            if self.Config.encoder.title() == 'Transformer' \
            else self.Config.emb_dim
        rnn_input_dim = emb_dim + 2 * self.Config.hidden_dim \
            if self.Config.attention_type == 'Bahdanau' and self.Config.encoder.title() == 'Recurrent' \
            else emb_dim

        self.rnn = getattr(nn, self.Config.rnn_type)(rnn_input_dim, model_dim, self.Config.num_layers, batch_first=True)
        self.attention = eval(f"{self.Config.attention_type}Attention({self.Config.emb_dim}, {self.Config.hidden_dim}, {self.Config.bias})")
        self.projector_dec2emb = nn.Linear(model_dim, emb_dim, bias=self.Config.bias)
        self.projector_2out = nn.Linear(emb_dim, self.vocab.__len__(), bias=self.Config.bias)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        # pointer network
        self.pointer = pointer

    def forward(self, h_j, dec_state, ys_prev, enc_idx=None):
        '''
        :param h_j: encoder annotations/outputs h_1, ..., h_Tx e.g.(batch_size, max_enc_len, 2 * hidden_dim) - (16, 400, 2 * 256)
        :param dec_state: final encoder hidden state, (num_layer, batch_size, 2*hidden_dim) - (2, 16, 2*256)
                or tuple of both final hidden and cell state (LSTM)
        :param ys_prev: decoder input sequence, (batch_size, max_dec_len) - (16, 80)
        :param enc_idx: only passed for pointer network - (16, 400)
        :return:
        '''
        bs = h_j.size(0)
        current_h_j = h_j
        current_enc_idx = enc_idx
        max_outputs = None

        s_t = dec_state  # entire final encoding state - hidden (plus cell) for all layers from num_layers
        if self.Config.encoder.title() == 'Transformer' and self.Config.rnn_type == 'LSTM':
            s_t = (s_t, s_t)
        if self.training:  # training mode, ys_prev not None
            ys_prev = self.embedding(ys_prev)  # (16, 80, 300) - for teacher forcing

        # time step zero, START_DEC-token, also contained as first seq element in dec_inputs dec_inputs[:, 0, :]
        # -> however in eval mode we do not have dec_inputs
        y_prev = np.ones((h_j.size(0),), dtype=int) * self.vocab[START_DEC]
        y_prev = torch.from_numpy(y_prev).to(self.device)
        y_prev = self.embedding(y_prev)

        outputs, weights = [], []

        if self.windower:
            current_h_j = h_j[:, 0:self.windower.ws, :]
            current_enc_idx = enc_idx[:, 0:self.windower.ws]
            enc_slider = EncoderSlider(h_j, enc_idx, self.windower)

        for t in range(self.dec_max_len):

            if self.windower:
                if t==0: max_outputs = torch.tensor(bs*[-1]) # dynamic inference: no move at t==0 (no decoder prediction)
                current_h_j, current_enc_idx = enc_slider.inference(current_h_j, current_enc_idx, max_outputs, self.vocab) \
                    if self.windower.type == 'dynamic' and not self.training \
                    else enc_slider.slide(current_h_j, current_enc_idx, t)

            dec_outputs, s_t, a_ij, max_outputs = self.one_step_decode(current_h_j, s_t, y_prev, enc_idx, current_enc_idx)

            weights.append(a_ij)
            outputs.append(dec_outputs)

            y_prev = self.embedding(max_outputs)  # (16, 300) - input for next step according to model
            tf_prob = self.tf_prob if self.training else None
            if tf_prob and tf_prob > random.uniform(0, 1) and t+1 < self.dec_max_len:  # teacher forcing
                y_prev = ys_prev[:, t+1, :]  # input from next label (teacher)


        w_ = torch.stack(weights)
        if len(w_.size()) < 3:
            w_ = w_.unsqueeze(1)
        # (16, 50000, 80), (16, 80, 400)
        return torch.stack(outputs).permute(1, 2, 0), w_.permute(1, 0, 2)

    def one_step_decode(self, h_j, dec_state, y_prev, enc_idx=None, current_enc_idx=None):
        '''
        one-step-decoding for beam search decoder
        :param h_j: encoder annotations/outputs h_1, ..., h_Tx e.g.(batch_size, max_enc_len, 2 * hidden_dim) - (16, 400, 2 * 256)
        :param dec_state: final encoder hidden state, (num_layer, batch_size, 2*hidden_dim) - (2, 16, 256)
                or tuple of both final hidden and cell state (LSTM)
        :param y_prev: decoder input index, (batch_size,) - (16,)
        :return:
        '''
        _last_h = lambda layers: layers[0][-1].unsqueeze(0) if isinstance(layers, tuple) else layers[-1].unsqueeze(
            0)  # extract only last layer

        s_t = dec_state  # entire final encoding state - hidden (plus cell) for all layers from num_layers
        dec_outputs, a_ij = None, None

        if self.Config.attention_type == 'Bahdanau':
            dec_h = _last_h(s_t)  # (1, 16, 512) - only last layer from num_layers for hidden state
            # dec_h = self.projector_dec2emb(dec_h)  # (1, 16, 300)
            a_ij, att_vector, _ = self.attention(h_j, dec_h, y_prev)  # (16, 400), (16, 2*256+300)
            dec_outputs, s_t = self.rnn(att_vector.unsqueeze(1), s_t)  # (16, 1, 2*256), ~
            dec_outputs = self.projector_dec2emb(dec_outputs).squeeze()  # (16, 300)

        elif self.Config.attention_type == 'Luong':

            if self.pointer:
                dec_h = _last_h(s_t)  # (1, 16, 512) - only last layer from num_layers for hidden state
                # dec_h = self.projector_dec2emb(dec_h)

            dec_outputs, s_t = self.rnn(y_prev.unsqueeze(1), s_t)  # (16, 1, 2*256), ~
            # dec_outputs = self.projector_dec2emb(dec_outputs)
            a_ij, att_vector = self.attention(h_j, dec_outputs)  # (16, 400), (16, 2*256)
            dec_outputs = att_vector
            dec_outputs = self.projector_dec2emb(dec_outputs)  # (16, 300)

        dec_outputs = self.projector_2out(dec_outputs)  # (16, 50000)
        dec_outputs = self.log_softmax(dec_outputs)  # (16, 50000)
        max_outputs = torch.argmax(dec_outputs, dim=-1)  # (16,)

        if self.pointer:  # pointing mechanism
            p_args = [dec_outputs, dec_h, y_prev, att_vector, a_ij,
                      enc_idx, current_enc_idx, self.vocab]
            dec_outputs, max_outputs = self.pointer(*p_args)

        if len(dec_outputs.size()) <= 1:
            dec_outputs = dec_outputs.unsqueeze(0)
        # (16, 50000), *, (16, 400), (16, 50005)
        return dec_outputs, s_t, a_ij, max_outputs

    def monte_carlo_sampling(self, h_j, dec_state, enc_idx=None):
        current_h_j = h_j
        current_enc_idx = enc_idx
        s_t = dec_state  # entire final encoding state - hidden (plus cell) for all layers from num_layers

        y_prev = np.ones((h_j.size(0),), dtype=int) * self.vocab[START_DEC]
        y_prev = torch.from_numpy(y_prev).to(self.device)
        y_prev = self.embedding(y_prev)

        ys, neg_log_probs, weights = [], [], []

        if self.windower:
            current_h_j = h_j[:, 0:self.windower.ws, :]
            current_enc_idx = enc_idx[:, 0:self.windower.ws]
            enc_slider = EncoderSlider(h_j, enc_idx, self.windower)

        for t in range(self.dec_max_len):

            if self.windower:
                current_h_j, current_enc_idx = enc_slider.slide(current_h_j, current_enc_idx, t)

            dec_outputs, s_t, a_ij, _ = self.one_step_decode(current_h_j, s_t, y_prev, enc_idx,
                                                                       current_enc_idx)
            sample_probs = torch.exp(dec_outputs)
            cat_dist = Categorical(sample_probs)
            sample = cat_dist.sample()
            ys.append(sample)
            neg_log_prob = -cat_dist.log_prob(sample)
            sample_masked = mask_oov(sample, self.vocab)
            y_prev = self.embedding(sample_masked)

            weights.append(a_ij)
            neg_log_probs.append(neg_log_prob)

        return torch.stack(neg_log_probs).transpose(0, 1), \
               torch.stack(ys).transpose(0, 1), \
               torch.stack(weights).transpose(0, 1)  # (batch_size, max_dec_len, max_enc_len)

    @classmethod
    def build(cls, Config, device, vocab):
        windower = None
        if Config.windowing:
            scheduler = StaticScheduler(Config.k, Config.d, Config) \
                if Config.w_type == "static" \
                else DynamicScheduler(vocab, None)
            windower = Windower(scheduler, Config, vocab)
        pointer = None
        if Config.pointer:
            pointer = PointerNet(Config, device)
        return Config, vocab, device, pointer, windower
