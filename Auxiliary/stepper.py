import torch
import torch.nn as nn
import numpy as np
import os
from constants import EMB_PATH, MODELDUMP_PATH, PAD
from Auxiliary.evaluation import pred2scores, merge_dicts
import time
from Auxiliary.utils import mask_oov, ids2sentence, make_readable, printScoresSelected
from data import Vocabulary
from Model.recurrent import EncoderBiGRU, EncoderBiLSTM, DecoderRNN
from Model.transformer import EncoderTransformer, DecoderTransformer
from Model.beam import BeamSearchDecoder

class Stepper:
    def __init__(self, Config, device):
        self.Config = Config
        self.device = device
        self.training = True
        self._step = 0
        self._lr = 0
        self.vocab, self.embedding = Setup.embedding(self.Config)
        self.lambda_ = self.Config.lambda_
        self.mle_criterion = eval(self.Config.criterion)
        setattr(self.mle_criterion, 'ignore_index', self.vocab[PAD])
        self.encoder, self.decoder =\
            Setup.encoder(self.Config, self.embedding, self.vocab, device),\
            Setup.decoder(self.Config, self.embedding, self.vocab, device),
        if self.Config.eval_beam:
            self.bsdecoder = Setup.beam_decoder(self.Config, self.vocab, self.decoder, device)
        if self.Config.tied_weights:
            self.decoder.projector_2out.weight = self.decoder.embedding.weight
        self.encoder.to(self.device); self.decoder.to(self.device)
        self.enc_optimizer, self.dec_optimizer = self._resume() \
            if self.Config.resume \
            else self._init_opt()

    def forward_mle(self, batch):
        n = batch.__len__()
        self.decoder.dec_max_len = batch.dec_max_len
        # remove OOV from enc_inputs for encoder, since no embeddings available for OOV
        enc_inputs_no_oov = mask_oov(batch.enc_inputs_oov, self.vocab)
        dec_targets_no_oov = mask_oov(batch.dec_targets_oov, self.vocab)

        if self.Config.encoder == 'Recurrent':
            enc_outputs, enc_state = self.encoder(enc_inputs_no_oov)
            dec_first_state = self.encoder.hidden_final(enc_state)
        else:  # Transformer
            enc_outputs = self.encoder(enc_inputs_no_oov)
            dec_first_state = self.encoder.hidden_final(enc_outputs)

        if self.Config.decoder == 'Recurrent':
            dec_outputs, att_weights = self.decoder(enc_outputs, dec_first_state, batch.dec_inputs, batch.enc_inputs_oov)
        else:
            dec_outputs, att_weights = self.decoder(enc_inputs_no_oov, enc_outputs, batch.dec_inputs)

        beam_dict = None
        if self.Config.eval_beam and not self.training:
            beam_batch_time = -time.time()
            self.bsdecoder.batch_size = n
            self.bsdecoder.dec_max_len = batch.dec_max_len
            beam_dec_outputs = self.bsdecoder(enc_outputs, dec_first_state, batch.enc_inputs_oov)
            beam_batch_time += time.time()
            beam_dict = {'beam_dec_outputs': beam_dec_outputs, 'beam_batch_time': beam_batch_time}

        targets = batch.dec_targets_oov if self.Config.pointer else dec_targets_no_oov
        loss = self.mle_criterion(dec_outputs, targets)

        return dec_outputs, beam_dict, loss, att_weights

    def forward_mixed(self, batch):
        assert self.Config.decoder == 'Recurrent', 'SCST not implemented for Transformer Decoder'
        n = batch.__len__()
        self.decoder.dec_max_len = batch.dec_max_len
        # remove OOV from enc_inputs for encoder, since no embeddings available for OOV
        enc_inputs_no_oov = mask_oov(batch.enc_inputs_oov, self.vocab)
        dec_targets_no_oov = mask_oov(batch.dec_targets_oov, self.vocab)

        if self.Config.encoder == 'Recurrent':
            enc_outputs, enc_state = self.encoder(enc_inputs_no_oov)
            dec_first_state = self.encoder.hidden_final(enc_state)
        else:  # Transformer
            enc_outputs = self.encoder(enc_inputs_no_oov)
            dec_first_state = self.encoder.hidden_final(enc_outputs)

        # TODO implement for Transformer Decoder
        # 1) MLE - training
        mle_outputs, mle_weights = self.decoder(enc_outputs, dec_first_state, batch.dec_inputs, batch.enc_inputs_oov)
        tf_prob = self.decoder.tf_prob
        self.decoder.tf_prob = 0  # set teacher forcing probability to 0 -> inference behavior (greedy decode)
        # 2) Greedy Decode -> (batch_size, max_vocab, max_dec_len), (batch_size, max_dec_len, max_enc_len)
        greedy_outputs, greedy_weights = self.decoder(enc_outputs, dec_first_state, batch.dec_inputs, batch.enc_inputs_oov)
        self.decoder.tf_prob = tf_prob  # back to teacher forcing settings
        # 3) Monte Carlo Sampling -> (batch_size, max_dec_len) * 2, (batch_size, max_dec_len, max_enc_len)
        mc_neg_log_probs, mc_ys, mc_weights = self.decoder.monte_carlo_sampling(enc_outputs, dec_first_state, batch.enc_inputs_oov)

        targets = batch.dec_targets_oov if self.Config.pointer else dec_targets_no_oov
        mle_loss = self.mle_criterion(mle_outputs, targets)

        # Rouge-L Reward for Policy Gradient Update
        greedy_preds = torch.argmax(greedy_outputs.transpose(1, 2), dim=-1).cpu().numpy()
        mc_preds = mc_ys.cpu().detach().numpy()
        greedy_reward = self._reward(batch, greedy_preds)
        mc_reward = self._reward(batch, mc_preds)

        final_reward = []
        for mr, gr in zip(mc_reward, greedy_reward):
            if mr == 0 or gr == 0:
                final_reward.append(0)
            else:
                final_reward.append(mr - gr)

        final_reward = torch.from_numpy(np.array(final_reward)).float().to(self.device)
        rl_loss = mc_neg_log_probs.mean(dim=-1) * final_reward
        rl_loss = rl_loss.mean()
        loss = self.Config.lambda_ * rl_loss + (1 - self.Config.lambda_) * mle_loss

        return loss

    def _reward(self, batch, preds):
        num_w = self._num_w(batch)
        system_summaries = [ids2sentence(tar, self.vocab, oov) for tar, oov in zip(batch.dec_targets_oov.cpu().numpy(), batch.encoder_oovs)]

        system_summaries = [make_readable(s, True, num_w[i]) for i, s in enumerate(system_summaries)]
        reward = pred2scores(batch, system_summaries, preds, self.vocab, num_w, batch_wise=False)

        return reward

    def _num_w(self, batch):
        if self.Config.windowing and self.Config.w_type == 'dynamic':
            num_w = [self.decoder.windower.scheduler.num_w(idx, self.Config.ws, self.Config.ss) for idx in batch.enc_inputs_oov.cpu().numpy()]
            return np.array(num_w)
        else:
            return np.array(len(batch)*[None])

    def predict(self, batch):
        self.eval()
        dec_outputs, beam_dict, loss, att_weights = self.forward_mle(batch)
        self.train()
        return dec_outputs, beam_dict, loss, att_weights

    def toy_output(self, batch, dec_outputs, beam_dict):

        article = ids2sentence(batch.enc_inputs_oov[0].cpu().numpy(), self.vocab, batch.encoder_oovs[0])
        summary = ids2sentence(batch.dec_targets_oov[0].cpu().numpy(), self.vocab, batch.encoder_oovs[0])

        dec_pred = torch.argmax(dec_outputs[0].transpose(0, 1), dim=1)  # log probs -> word idx
        pred = ids2sentence(dec_pred.cpu().detach().numpy(), self.vocab, batch.encoder_oovs[0])
        beam_pred = None
        if beam_dict:
            beam_pred = ids2sentence(beam_dict['beam_dec_outputs'][0][0].cpu().detach().numpy(), self.vocab, batch.encoder_oovs[0])

        print("Article: " + article)
        print("OOV: " + "|".join(batch.encoder_oovs[0]))
        print("Summary: " + summary)
        print("Predicted Greedy Search: " + pred)
        if beam_pred:
            print("Predicted Beam Search " + beam_pred)

    def metric_scores(self, batch, dec_outputs, beam_dict, with_meteor=False):

        num_w = self._num_w(batch)
        system_summaries = [ids2sentence(tar, self.vocab, oov) for tar, oov in zip(batch.dec_targets_oov.cpu().numpy(), batch.encoder_oovs)]
        system_summaries = [make_readable(s, True, num_w[i]) for i, s in enumerate(system_summaries)]

        metric_dict = None
        if dec_outputs is not None:
            dec_preds = torch.argmax(dec_outputs.transpose(1, 2), dim=-1).cpu().numpy()
            metric_dict = pred2scores(batch, system_summaries, dec_preds, self.vocab, self._num_w(batch), with_meteor=with_meteor)

        beam_metric_dict = None
        if beam_dict:
            dec_preds_beam = beam_dict['beam_dec_outputs'][:, 0].cpu().numpy()  # 0 - only best hyp
            beam_metric_dict = pred2scores(batch, system_summaries, dec_preds_beam, self.vocab, self._num_w(batch), with_meteor=with_meteor)

        return metric_dict, beam_metric_dict

    def metric_output(self, rouge_val_dicts, rouge_beam_val_dicts, count, beam_count, meteor=None, beam_meteor=None):
        print("==========================ROUGE=======================")
        if rouge_val_dicts:
            final_rouge_dict = merge_dicts(rouge_val_dicts, count)
            print("==SCORES-GREEDY==")
            printScoresSelected(final_rouge_dict, meteor)
        if rouge_beam_val_dicts:
            final_beam_rouge_dict = merge_dicts(rouge_beam_val_dicts, beam_count)
            print("==SCORES-BEAM==")
            printScoresSelected(final_beam_rouge_dict, beam_meteor)
        print("======================================================")

    def backward(self, loss):
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()
        loss.backward()

    def opt_step(self):
        self._step += 1
        self._anneal()
        for p in self.enc_optimizer.param_groups:
            p['lr'] = self._lr
        for p in self.dec_optimizer.param_groups:
            p['lr'] = self._lr
        self.enc_optimizer.step()
        self.dec_optimizer.step()

    def eval(self):
        self.training = False
        self.encoder.eval()
        self.decoder.eval()

    def train(self):
        self.training = True
        self.encoder.train()
        self.decoder.train()

    def _init_opt(self):
        enc_params = self.encoder.parameters()
        enc_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, enc_params), lr=0,
                                         betas=(self.Config.beta_1, self.Config.beta_2), eps=float(self.Config.opt_eps),
                                              weight_decay=float(self.Config.weight_decay))
        dec_params = self.decoder.parameters()
        dec_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dec_params), lr=0,
                                         betas=(self.Config.beta_1, self.Config.beta_2), eps=float(self.Config.opt_eps),
                                         weight_decay=float(self.Config.weight_decay))
        return enc_optimizer, dec_optimizer

    def save(self, epoch):
        state = {'Config': self.Config,
                 'encoder': self.encoder.state_dict(),
                 'decoder': self.decoder.state_dict(),
                 'enc_optimizer': self.enc_optimizer.state_dict(),
                 'dec_optimizer': self.dec_optimizer.state_dict(),
                 'lr': self._lr,
                 'step': self._step,
                 }

        save_path = os.path.join(MODELDUMP_PATH, f"model_{self.Config.save_description}_e{epoch}.pt")
        torch.save(state, save_path)

    @classmethod
    def load(self, from_save_description, from_epoch, device):
        load_path = os.path.join(MODELDUMP_PATH,
                                 f"model_{from_save_description}_e{from_epoch}.pt")
        print(f"Loading from {load_path}")
        state = torch.load(load_path, map_location=device)
        return state

    def _resume(self):
        state = self.load(self.Config.from_save_description, self.Config.from_epoch, self.Config.device)
        self._step = state['step']
        self._lr = state['lr']
        self.encoder.load_state_dict(state['encoder'])
        self.decoder.load_state_dict(state['decoder'])
        enc_optimizer, dec_optimizer = self._init_opt()
        enc_optimizer.load_state_dict(state['enc_optimizer'])
        dec_optimizer.load_state_dict(state['dec_optimizer'])

        return enc_optimizer, dec_optimizer

    def _anneal(self):
        model_dim = self.Config.d_model \
            if self.Config.encoder == 'Transformer' or self.Config.decoder == 'Transformer' \
            else self.Config.emb_dim
        self._lr = self.Config.factor * model_dim ** (-0.45) * min(self._step ** (-0.5), self._step * self.Config.warmup ** (-1.5))

class Setup:
    @classmethod
    def embedding(cls, Config):
        emb_dim = Config.emb_dim if Config.encoder == 'Recurrent' else Config.d_model
        embedding = nn.Embedding(Config.max_vocab, emb_dim)

        if Config.init:  # initialize embeddings with pre-dumped Fasttext embeddings
            try:
                embedding.load_state_dict(torch.load(os.path.join(os.path.dirname(EMB_PATH),
                                                                  f"dump/initial_{Config.dataset}_{Config.max_vocab}_{Config.emb_dim}.pt")))
            except:
                raise Exception(f"First store an embedding file \
                        'initial_{Config.dataset}_{Config.max_vocab}_{Config.emb_dim}.pt' under embeddings/dump/")

        # should always be trainable, because considerable number of embeddings is initialized randomly
        embedding.weight.requires_grad = Config.trainable
        vocab = Vocabulary(Config.vocab_path, Config.max_vocab)
        return vocab, embedding

    @classmethod
    def encoder(cls, Config, embedding, vocab, device):
        if Config.encoder.title() == "Recurrent":
            if Config.rnn_type.upper() == "LSTM":
                encoder = EncoderBiLSTM(
                    Config=Config,
                    embedding=embedding,
                    device=device
                )
            else:  # GRU
                encoder = EncoderBiGRU(
                    Config=Config,
                    embedding=embedding,
                    device=device
                )
        else:
            encoder = EncoderTransformer(
                embedding, Config.N, vocab,
                *EncoderTransformer.build(d_model=Config.d_model,
                                          d_ff=Config.d_ff,
                                          h=Config.h,
                                          dropout=Config.dropout)
            )
        return encoder

    @classmethod
    def decoder(cls, Config, embedding, vocab, device):
        if Config.decoder.title() == 'Recurrent':
            decoder = DecoderRNN(
                embedding, *DecoderRNN.build(Config=Config,
                                             device=device,
                                             vocab=vocab)
            )

        else:
            decoder = DecoderTransformer(
                embedding, Config.N, vocab, device, Config.dec_max_len, None,
                *DecoderTransformer.build(d_model=Config.d_model,
                                          d_ff=Config.d_ff,
                                          h=Config.h,
                                          dropout=Config.dropout)
            )
        return decoder


    @classmethod
    def beam_decoder(cls, Config, vocab, decoder, device):
        bsdecoder = BeamSearchDecoder(
            Config=Config,
            vocab=vocab,
            decoder=decoder,
            device=device
            )
        return bsdecoder







