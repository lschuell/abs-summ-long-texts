import math
from constants import PAD, STOP_DEC, DOT
from sklearn.preprocessing import normalize
from Auxiliary.utils import ids2sentence, make_readable
import numpy as np

class Scheduler(object):
    def __call__(self, *args):
        raise NotImplementedError

class DynamicScheduler(Scheduler):
    def __init__(self, vocab, word2vec):
        self.vocab = vocab
        self.word2vec = word2vec

    def __call__(self, dec_inputs):
        '''assumes batch has been adjusted by self.batchup'''
        return (dec_inputs == self.vocab[STOP_DEC]).nonzero().view(-1).cpu().numpy()

    def inst_up(self, inst, ws, ss, enc_max_len, dec_max_len):
        '''adjust batch object to dynamic scheduler'''
        map_ = self._map2win(inst, ws, ss)
        diff_ = np.ediff1d(map_)
        decoder_input = self._add_eos(inst.decoder_input, diff_, inst.dec_max_len)
        decoder_target_pointer = self._add_eos(inst.decoder_target_pointer, diff_, inst.dec_max_len)
        max_w = math.ceil((enc_max_len - ws)/ss) + 1
        _end_pad = lambda x: np.pad(x, (0, (dec_max_len + max_w) - len(x)), "constant",
                                    constant_values=(self.vocab[PAD],))
        dml = len(decoder_input)
        decoder_input = _end_pad(decoder_input)
        decoder_target_pointer = _end_pad(decoder_target_pointer)

        return dml, decoder_input, decoder_target_pointer

    def _pos2win(self, start, end, ws, ss, num_w):
        '''maps position - (start, end) to -> window number'''
        b_vals = []
        no = []
        for i in range(num_w):
            w_start = i * ss
            w_end = ws + i * ss
            start_in = w_start <= start <= w_end
            end_in = w_start <= end <= w_end
            if start_in and end_in:
                return i
            if start_in:
                b_vals.append(w_end - start)
                no.append(i)
            if end_in:
                b_vals.append(end - w_start)
                no.append(i)
        return no[np.argmin(b_vals)]

    def _aggregate(self, ss_):
        sent = ss_.split(" ")
        vecs = [self.word2vec[w] for w in sent if w in self.word2vec]
        if vecs:
            vecs = normalize(np.stack(vecs), axis=1, norm='l2')
            mean = np.mean(vecs, 0)
            return mean.squeeze()
        return None


    def _map2win(self, inst, ws, ss):
        '''list - for each element in the batch, derive np.array of sequentialized window positions'''
        summary = ids2sentence(inst.decoder_target_pointer, self.vocab, inst.encoder_oovs)
        summary = make_readable(summary, True)
        article = ids2sentence(inst.encoder_pointer_idx, self.vocab, inst.encoder_oovs)
        article = make_readable(article, True)
        num_w = self.num_w(inst.encoder_pointer_idx, ws, ss)
        art_sents = article.split(" . ")
        sum_sents = summary.split(" . ")
        scores = []
        for ss_ in sum_sents:
            for as_ in art_sents:
                vec_ss_ = self._aggregate(ss_)
                vec_as_ = self._aggregate(as_)
                score = vec_ss_ @ vec_as_ \
                    if vec_ss_ is not None and vec_as_ is not None \
                    else 0
                scores.append(score)
                '''
                try:
                    _, rouge_dict, _ = setup_and_eval([as_ + " ."], [ss_ + " ."])
                    scores.append(rouge_dict['rouge_l_recall'])
                except:
                    scores.append(0)
                '''
        scores = np.array(scores).reshape(len(sum_sents), -1)
        ss2win = []
        for ss_ in scores:
            as_ = np.argmax(ss_)
            start = " . ".join(art_sents[:as_]).split(" ").__len__() + 1
            end = art_sents[as_].split(" ").__len__() + start
            win = self._pos2win(start, end, ws, ss, num_w)
            ss2win.append(win)
        sequential_ss2win = np.maximum.accumulate(ss2win)
        return sequential_ss2win

    '''
    def _pad(self, decoder_idx):
        #unfiy different example-lengths due to different number of window-slide steps (EOS)
        max_ = 0
        for ex in decoder_idx:
            max_ = max(max_, len(ex))

        for i, t in enumerate(decoder_idx):
            to_pad = max_ - len(t)
            if to_pad > 0:
                decoder_idx[i] = torch.cat([decoder_idx[i], torch.tensor(to_pad * [self.vocab[PAD]])])

        return max_, torch.stack(decoder_idx)
    '''

    def _add_eos(self, decoder_idx, nums_eos, dec_max_len):
        '''insert EOS-tokens into decoder sequence, where nums_eos specifies number
        of EOS-tokens to insert at each summary sentence end'''
        dec_idx = decoder_idx[:dec_max_len]
        pos = (dec_idx == self.vocab[DOT]).nonzero()[0]
        if pos.size != 0:
            sents = list(np.split(dec_idx, pos + 1))
            sents = [arr for arr in sents if arr.size != 0]
            for j, num_eos in enumerate(nums_eos):
                if num_eos > 0:
                    sents[j] = np.concatenate((sents[j], np.array(num_eos * [self.vocab[STOP_DEC]])))
            return np.concatenate(sents)
        else:
            return dec_idx

    def num_w(self, encoder_idx, ws, ss):
        '''derive maximum number of windows for given encoder sequence'''
        length = (encoder_idx != self.vocab[PAD]).sum().item()
        remainder = length - ws
        rest = 1 if remainder % ss == 0 \
            else 2
        num_w = remainder // ss + rest if remainder > 0 else 1
        return num_w

class StaticScheduler(Scheduler):
    def __init__(self, *args):
        '''
        :param k: first skewness parameter
        :param d: second skewness parameter
        :param config:
        '''
        self.k, self.d, self.Config = args

    def __call__(self, num_w, len, last_p):
        '''
        :param num_w: number of windows
        :param len: length of input sequence
        :param last_p: proportion of relevant information in last window
        :return: sequence of positions where window moves
        '''
        dist = self._vec_exp_decay(np.arange(num_w))
        dist[-1] *= last_p
        dist = dist / dist.sum()
        expected_sum_len = int(len / self.Config.max_corpus_len * self.Config.dec_max_len)
        return (dist.cumsum() * expected_sum_len).astype(int)[:-1]

    def _vec_exp_decay(self, t):
        return np.vectorize(self._exp_decay)(t)

    def _exp_decay(self, t):
        return np.exp(-self.k * (1 + t * self.d ** t))


class Windower(object):
    def __init__(self, scheduler, Config, vocab):
        self.scheduler = scheduler
        c_n = self.scheduler.__class__.__name__.lower()
        self.type = c_n[:c_n.find('scheduler')]
        self.vocab = vocab
        self.ws = Config.ws  # window size
        self.ss = Config.ss  # step size
        self.max_window = math.ceil((Config.enc_max_len - self.ws)/self.ss) + 1

    def __call__(self, idx):
        return self._get_schedule(idx)

    def _get_schedule(self, idx):
        if self.type == 'dynamic': # idx=dec_inputs
            return self.scheduler(idx)
        else:  # 'static', idx=encoder_idx
            length = (idx != self.vocab[PAD]).sum().item()
            '''
            remainder = length - self.ws
            last_p = (remainder % self.ss) / self.ss if remainder > 0 else 1
            num_w = remainder // self.ss + 2 if remainder > 0 else 1
            '''
            remainder = length - self.ws
            rest = 1 if remainder % self.ss == 0 \
                else 2
            last_p = (remainder % self.ss) / self.ss if remainder > 0 else 1
            num_w = remainder // self.ss + rest if remainder > 0 else 1
            return self.scheduler(num_w, length, last_p)

    def __len__(self):
        return self.max_window


class EncoderSlider(object):
    def __init__(self, h_j, enc_idx, windower):
        '''
        :param h_j: encoder annotations/outputs h_1, ..., h_Tx e.g.(batch_size, max_enc_len, 2 * hidden_dim) - (16, 400, 2 * 256)
        :param enc_idx:
        :param windower: Windows object
        '''
        self.bs = h_j.__len__()
        self.h_j = h_j
        self.enc_idx = enc_idx
        self.windower = windower
        self.at = np.zeros(self.bs, dtype=int)
        self.transitions, self.num_windows = self._setup()
        self.num_w = None
        if self.windower.type == 'dynamic':
            self.num_w = [self.windower.scheduler.num_w(idx, self.windower.ws, self.windower.ss) for idx in self.enc_idx.cpu().numpy()]
            self.num_w = np.array(self.num_w)

    def _setup(self):
        '''
        Setup transitions and num_windows lookup tables
        :return transitions, e.g. batch size = 4 : [[60, 72, 80, 95], [], [63, 90], [80]]
                             filled to np.array([[60,72,80,95],[-1,-1,-1,-1],[63,90,-1,-1],[80,-1,-1,-1]])
                num_windows, e.g. batch size = 4 : [4, 0, 2, 1]
        '''
        transitions = [list(self.windower(idx)) for idx in self.enc_idx]
        num_windows = [t.__len__() for t in transitions]
        transitions = np.array([(t + [-1] * (self.windower.__len__() - 1 - len(t)))[:(self.windower.__len__() - 1)] for t in transitions])
        return transitions, num_windows

    def inference(self, win_h_j, win_enc_idx, *args):
        '''Inference mode for dynamic windowing, if decoder ejects EOS, window is pushed ahead'''
        max_outputs, vocab = args
        if np.any(max_outputs.cpu().numpy() == vocab[STOP_DEC]):
            up_idx = np.where(max_outputs.cpu().numpy() == vocab[STOP_DEC])[0]

            ws, ss = self.windower.ws, self.windower.ss
            new_win_h_j, new_win_enc_idx = win_h_j.clone(), win_enc_idx.clone()
            for up in up_idx:
                if self.at[up] == self.num_w[up] - 1: continue
                self.at[up] += 1
                current_win = self.at[up]
                new_win_h_j[up] = self.h_j[up, ss * current_win:ss * current_win + ws, :]
                new_win_enc_idx[up] = self.enc_idx[up, ss * current_win:ss * current_win + ws]

            return new_win_h_j, new_win_enc_idx

        return win_h_j, win_enc_idx

    def slide(self, win_h_j, win_enc_idx, t):
        '''
        update windowed encoder annotations used for attention
        :param win_h_j: windowed encoder annotations/outputs h_1, ..., h_Tx e.g. (batch_size, ws, 2 * hidden_dim) - (16, 400, 2 * 256)
        :param win_enc_idx: windowed encoder idx
        :param t: decoding time step
        :param max_outputs: only passed for dynamic scheduler at inference time
        '''

        if np.any(self.transitions >= 0) and np.any(self.transitions[:, 0] == t):
            #up_idx = np.where(self.transitions[:, 0] == t)[0]
            # consider accumulation of transition indices
            up_idx = np.where((0 <= self.transitions[:, 0]) & (self.transitions[:, 0] <= t))[0]

            ws, ss = self.windower.ws, self.windower.ss
            new_win_h_j, new_win_enc_idx = win_h_j.clone(), win_enc_idx.clone()
            for up in up_idx:

                self.at[up] += 1
                current_win = self.at[up]
                new_win_h_j[up] = self.h_j[up, ss * current_win:ss * current_win + ws, :]
                new_win_enc_idx[up] = self.enc_idx[up, ss * current_win:ss * current_win + ws]
                self.transitions[up][:-1] = self.transitions[up][1:]
                self.transitions[up][-1] = -1

            return new_win_h_j, new_win_enc_idx

        return win_h_j, win_enc_idx

