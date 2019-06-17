import torch
import numpy as np
from constants import START_DEC, STOP_DEC
from Auxiliary.utils import mask_oov
from Model.windowing import EncoderSlider


class BeamSearchDecoder(object):
    def __init__(self, Config, vocab, decoder, device):
        self.Config = Config
        self.vocab = vocab
        self.sos, self.eos = self.vocab[START_DEC], self.vocab[STOP_DEC]
        self.decoder = decoder
        self.device = device
        self.enc_dim = self.decoder.windower.ws if self.decoder.windower else self.Config.enc_max_len
        self.B = self.Config.B
        self.batch_size = self.Config.batch_size
        self.dec_max_len = None  # attribute is set on the fly according to batch max_dec_len
        self.hidden_dim = 2 * self.Config.hidden_dim if self.Config.decoder == 'Recurrent' else self.Config.d_model
        self.hyps = None  # B hypotheses with word idx for each instance
        self.hyps_pointer = None  # B hypotheses in pointer model
        self.beam_lens = None  # current sequence lengths
        self.avg_log_probs = None  # average log probability of hyps
        self.eos_mask = None  # indicator table whether hyp reached EOS-token
        self.enc_outs = None  # expanded encoder output lookup
        self.current_h_j = None # current window of encoder output
        self.current_enc_idx = None
        self.dec_states = None  # expanded decoder state lookup
        self.enc_slider = None
        self.num_w = None  # track number of possible windows in case of dynamic scheduler

    def set_windower(self, windower):
        '''
        change windower object and related attributes such as enc_dim on the fly
        '''
        self.decoder.windower = windower
        self.enc_dim = self.decoder.windower.ws \
            if windower \
            else self.Config.max_enc_len

    def _expand_dec_s(self, dec_s):
        '''
        expands decoder state into lookup object
        '''
        _expand = lambda dec_s: dec_s.transpose(0, 1).repeat(1, self.B, 1) \
            .view(-1, self.hidden_dim).unsqueeze(0)

        if isinstance(dec_s, tuple):  # LSTM
            return tuple([_expand(s) for s in dec_s])
        else:  # GRU
            return _expand(dec_s)

    def _update_lookup(self, lookup, update_idx):
        '''
        updates lookup object by shuffling/indexing along dimension of batch_size * B according to update_idx
        :param lookup: Lookup object, e.g. encoder outputs enc_outs/h_j (batch_size * B, enc_max_len, hidden_dim)
        :param update_idx: (batch_size * B) - e.g. [1, 1, 2, 3, 4, 5, 6, 7, 6, 6, ....]
        :return:
        '''

        if isinstance(lookup, tuple):
            dim = (torch.tensor(lookup[0].size()) == update_idx.size(0)).nonzero()[0].item()
            return tuple([torch.index_select(lp, dim, update_idx) for lp in lookup])
        else:
            dim = (torch.tensor(lookup.size()) == update_idx.size(0)).nonzero()[0].item()
            return torch.index_select(lookup, dim, update_idx)

    def _first_hyps(self, first_dec_out, first_dec_s):
        self.enc_idx = self.enc_idx.unsqueeze(1) \
            .repeat(1, self.B, 1).view(-1, self.Config.enc_max_len)
        self.current_enc_idx = self.current_enc_idx.unsqueeze(1) \
            .repeat(1, self.B, 1).view(-1, self.enc_dim)  # (16*5, 400)

        if self.Config.pointer:
            top_lprob, top_idx = torch.topk(first_dec_out, k=self.B, dim=1)
            self.hyps_pointer = top_idx.unsqueeze(2)  # (16, 5, 1)
            top_idx = mask_oov(top_idx, self.vocab)
        else:
            top_lprob, top_idx = torch.topk(first_dec_out, k=self.B, dim=1)

        self.hyps = top_idx.unsqueeze(2)  # (16, 5, 1)
        self.avg_log_probs = top_lprob  # (16, 5)
        self.beam_lens = torch.ones((self.batch_size, self.B), device=self.device)  # (16, 5)

        if self.decoder.windower and self.Config.w_type == 'dynamic':
            self.eos_mask = torch.from_numpy(self.num_w).view(self.batch_size, self.B).to(self.device)
            self.eos_mask = torch.where(top_idx == self.eos, self.eos_mask - 1, self.eos_mask)
            self.eos_mask[self.eos_mask < 0] = 0
        else:
            self.eos_mask = torch.ones((self.batch_size, self.B), dtype=torch.int, device=self.device)  # (16, 5)
            self.eos_mask = torch.where(top_idx == self.eos,
                                        torch.zeros((self.batch_size, self.B), dtype=torch.int, device=self.device),
                                       self.eos_mask)

        self.current_h_j = self.current_h_j.unsqueeze(1). \
            repeat(1, self.B, 1, 1).view(-1, self.enc_dim, self.hidden_dim)

        if self.Config.decoder.title() == 'Recurrent':
            self.dec_states = self._expand_dec_s(first_dec_s)  # (tuple of) (1, 80, 512)

    def _update_hyps(self):

        if self.Config.decoder.title() == 'Recurrent':
            dec_input = self.hyps[:, :, -1].view(-1)  # (batch_size * B,)
            dec_input = self.decoder.embedding(dec_input)
            next_dec_out, dec_s, _, _ = self.decoder.one_step_decode(self.current_h_j,
                                     self.dec_states, dec_input, self.enc_idx, self.current_enc_idx)  # (80, 50000)
        else:  # Transformer
            len_hyps = self.hyps.size(-1)
            dec_input = self.hyps.view(-1, len_hyps)  # (batch_size * B, len_hyps)
            next_dec_out, _ = self.decoder.one_step_decode(self.current_enc_idx,
                              self.current_h_j, dec_input)  # (80, 50000)


        top_lprob, top_idx = torch.topk(next_dec_out, k=self.B, dim=1)  # (80, 5)

        _repeat = lambda x: x.unsqueeze(2).repeat(1, 1, self.B)
        self.beam_lens += 1
        self.beam_lens = torch.where(self.eos_mask == 0, self.beam_lens - 1, self.beam_lens)
        # update mean average log probabilities normalized by sequence length

        new_avg_lprobs = (1 - 1. / _repeat(self.beam_lens)) * _repeat(self.avg_log_probs) + \
                         (1. / _repeat(self.beam_lens)) * top_lprob.view(self.batch_size, self.B, self.B)

        tmp_eos_mask = self.eos_mask.unsqueeze(2).repeat(1, 1, self.B)
        #new_avg_lprobs = torch.where(tmp_eos_mask == 1, new_avg_lprobs,
        #                             self.avg_log_probs.unsqueeze(2).repeat(1, 1, self.B))
        new_avg_lprobs = torch.where(tmp_eos_mask == 0, self.avg_log_probs.unsqueeze(2).repeat(1,1,self.B), new_avg_lprobs)

        #tmp_eos_mask[:, :, 0] = 1
        #new_avg_lprobs = torch.where(tmp_eos_mask == 1, new_avg_lprobs,
        #                             torch.ones((self.batch_size, self.B, self.B),
        #                             device=self.device).fill_(-100))

        # set all but one very low so that only one eos beam branch survives
        tmp_mask = tmp_eos_mask==0
        tmp_mask[:,:,:-1] = 0
        tmp_eos_mask = tmp_eos_mask.masked_scatter_(tmp_mask, tmp_eos_mask.clone().fill_(1))
        new_avg_lprobs = torch.where(tmp_eos_mask == 0, torch.ones((self.batch_size, self.B, self.B), device=self.device).fill_(-100), new_avg_lprobs)

        topB_avglprobs, topB_avgidx = new_avg_lprobs.view(self.batch_size, -1).topk(k=self.B, dim=1)
        per_inst_pos = topB_avgidx / self.B
        update_idx = (torch.arange(self.batch_size, dtype=torch.long, device=self.device)
                      .unsqueeze(1) * self.B + per_inst_pos).view(-1)

        if self.Config.pointer:
            next_dec_idx_p = torch.gather(top_idx.view(self.batch_size, -1), 1, topB_avgidx)  # indices
            self.hyps_pointer = torch.cat(
                (self._update_lookup(self.hyps_pointer.view(self.batch_size * self.B, -1), update_idx)
                 .view(self.batch_size, self.B, self.hyps_pointer.size(2)),
                 next_dec_idx_p.unsqueeze(2)), 2)
            top_idx = mask_oov(top_idx, self.vocab)

        next_dec_idx = torch.gather(top_idx.view(self.batch_size, -1), 1, topB_avgidx)  # indices

        if self.Config.decoder.title() == 'Recurrent':
            self.dec_states = self._update_lookup(dec_s, update_idx)

        self.hyps = torch.cat(
            (self._update_lookup(self.hyps.view(self.batch_size * self.B, -1), update_idx)
             .view(self.batch_size, self.B, self.hyps.size(2)),
              next_dec_idx.unsqueeze(2)), 2)

        self.avg_log_probs = topB_avglprobs
        self.eos_mask = self._update_lookup(self.eos_mask.view(-1), update_idx).view(self.batch_size, self.B)
        #self.eos_mask = torch.where(next_dec_idx == self.eos,
        #                            torch.zeros((self.batch_size, self.B), dtype=torch.int, device=self.device),
        #                           self.eos_mask)
        self.eos_mask = torch.where(next_dec_idx == self.eos, self.eos_mask - 1, self.eos_mask)
        self.eos_mask[self.eos_mask < 0] = 0
        self.beam_lens = self._update_lookup(self.beam_lens.view(-1), update_idx).view(self.batch_size, self.B)

        if self.decoder.windower and self.Config.w_type == 'dynamic':
            self.current_h_j = self._update_lookup(self.current_h_j, update_idx)
            self.current_enc_idx = self._update_lookup(self.current_enc_idx, update_idx)
            at = self._update_lookup(torch.from_numpy(self.enc_slider.at).to(self.device), update_idx)
            self.enc_slider.at = at.cpu().numpy()

    def __call__(self, h_j, dec_state, enc_idx=None):

        with torch.no_grad():
            self.enc_idx = enc_idx
            self.current_h_j = h_j
            self.current_enc_idx = enc_idx
            if self.decoder.windower:
                self.current_h_j = h_j[:, 0:self.decoder.windower.ws, :]
                self.current_enc_idx = enc_idx[:, 0:self.decoder.windower.ws]
                if self.Config.w_type == 'dynamic':  # dynamic sliding may lead to different h_j/enc_idx per hyp
                    h_j, enc_idx = h_j.unsqueeze(1).repeat(1, self.B, 1, 1), enc_idx.unsqueeze(1).repeat(1, self.B, 1)
                    h_j, enc_idx = h_j.view(-1, *h_j.size()[-2:]), enc_idx.view(-1, enc_idx.size(-1))
                self.enc_slider = EncoderSlider(h_j, enc_idx, self.decoder.windower)
                self.num_w = self.enc_slider.num_w

            dec_s = None
            if self.Config.decoder.title() == "Recurrent":
                dec_s = dec_state
                if self.Config.encoder.title() == 'Transformer' and self.Config.rnn_type == 'LSTM':
                    dec_s = (dec_s, dec_s)
                dec_input = np.ones((self.batch_size,), dtype=int) * self.sos
                dec_input = torch.from_numpy(dec_input).to(self.device)
                dec_input = self.decoder.embedding(dec_input)
                first_dec_out, dec_s, _, _ = self.decoder.one_step_decode(self.current_h_j,
                                          dec_s, dec_input, self.enc_idx, self.current_enc_idx)

            else:  # Transformer
                dec_input = np.ones((self.batch_size, 1), dtype=int) * self.sos
                dec_input = torch.from_numpy(dec_input).to(self.device)
                first_dec_out, _ = self.decoder.one_step_decode(self.current_enc_idx,
                                                                self.current_h_j, dec_input)

            self._first_hyps(first_dec_out, dec_s)

            step = 1

            slide_idx = torch.arange(self.batch_size) * self.B

            while self.eos_mask.sum().item() > 0 and step < self.dec_max_len:

                if self.decoder.windower:
                    if self.Config.w_type == 'dynamic':
                        max_outputs = self.hyps[:, :, -1].view(-1)
                        self.current_h_j, self.current_enc_idx = self.enc_slider\
                            .inference(self.current_h_j, self.current_enc_idx, max_outputs, self.vocab)
                    else:
                        self.current_h_j, self.current_enc_idx = self.enc_slider\
                            .slide(self.current_h_j[slide_idx], self.current_enc_idx[slide_idx], step)
                        self.current_h_j = self.current_h_j.unsqueeze(1).repeat(1, self.B, 1, 1)\
                            .view(-1, self.enc_dim, self.hidden_dim)
                        self.current_enc_idx = self.current_enc_idx.unsqueeze(1)\
                            .repeat(1, self.B, 1).view(-1, self.enc_dim)  # (16*5, 400)

                self._update_hyps()
                step += 1

        return self.hyps_pointer if self.Config.pointer else self.hyps

