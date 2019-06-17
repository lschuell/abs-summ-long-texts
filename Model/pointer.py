import torch
import torch.nn as nn
from Auxiliary.utils import mask_oov

class PointerNet(nn.Module):
    def __init__(self, Config, device):
        super(PointerNet, self).__init__()
        self.Config = Config
        self.device = device
        p_input_dim = 4 * self.Config.hidden_dim + self.Config.emb_dim
        if self.Config.attention_type.title() == 'Bahdanau':
            p_input_dim += self.Config.emb_dim
        self.project_ptr2prob = nn.Linear(p_input_dim, 1, bias=self.Config.bias)
        self.ptr_prob = nn.Sigmoid()

    def forward(self, *args):

        dec_outputs, dec_h, y_prev, \
        att_vector, a_ij, \
        enc_idx, current_enc_idx, vocab = args

        bs = y_prev.size(0)
        if bs == 1:
            att_vector = att_vector.unsqueeze(0)

        prob_ptr = \
            self.ptr_prob(self.project_ptr2prob(
                torch.cat((att_vector, dec_h.squeeze(0), y_prev), dim=-1)
            ))  # pointing probability

        prob_gen = 1 - prob_ptr  # generation probability
        max_oov_idx = enc_idx.max().item()  # maximum OOV index, e.g. 50006 = at most 7 OOV per document
        dec_outputs = prob_gen * torch.exp(dec_outputs)
        att_dist = prob_ptr * a_ij

        if max_oov_idx >= vocab.__len__():  # else no OOV, extending probability matrix not necessary
            dec_outputs = torch.cat((dec_outputs,
                                     torch.zeros(current_enc_idx.size(0),
                                                 max_oov_idx - vocab.__len__() + 1,
                                                 device=self.device)), 1)

        dec_outputs = dec_outputs.scatter_add(1, current_enc_idx, att_dist)  # add pointing probs to generation probs
        max_outputs = torch.argmax(dec_outputs, dim=1)
        max_outputs = mask_oov(max_outputs, vocab)
        dec_outputs = torch.log(dec_outputs + float(self.Config.eps))  # transform back to log_probs for NLLL

        return dec_outputs, max_outputs


