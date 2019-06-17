import sys
import pickle
import ast
import numpy as np
import torch
import copy
import re
from torch import nn
from constants import UNKNOWN, PAD, STOP_DEC


def mask_oov(data, vocab):
    '''
    mask out temporary oov ids by replacement with unknown token id
    :param data: data tensor with temporary oov ids, e.g. [126, 320, 12230, 50008, ..., 50002]
    :param vocab: vocabulary object, e.g. with size=50000
    :return: data tensor with temporary oov ids replaced by unknown id, e.g. [126, 320, 12230, 0, ..., 0]
    '''

    return torch.where(data < vocab.__len__(), data, torch.ones_like(data).fill_(vocab[UNKNOWN]))

def pad_mask(data, vocab):
    '''
    yield a mask indicating padding positions
    '''
    return (data != vocab[PAD]).unsqueeze(-2)

def future_mask(data, device):
    '''
    yield a mask indicating future positions
    '''
    size = data.size(-1)
    return (torch.triu(torch.ones((size, size), dtype=torch.int, device=device), diagonal=1) == 0).unsqueeze(0)

def future_and_pad_mask(data, vocab, device):
    '''
    yield a mask merging pad_mask and future_mask
    '''
    pm = pad_mask(data, vocab)  # (batch_size, 1, max_dec_len)
    fm = future_mask(data, device)  # (1, max_dec_len, max_dec_len)
    return pm & fm  # (batch_size, max_dec_len, max_dec_len)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def ids2sentence(test_example, vocab, decoded_oovs=None):
    '''
    :param test_example: [201, 2400, 500,...,40]
    :param vocab.id2word: {201:"man", 202:"woman",...,2400:"walk"}
    :param decoded_oovs: ["leifman", "hybrid-search", "pollination"] list of OOV tokens if none [""]
    :return: "man walk toward the street lantern"
    '''

    start_extend_idx = vocab.__len__()
    id2word = vocab.id2word.copy()

    #print(decoded_oovs, decoded_oovs.shape)
    if decoded_oovs is not None:  #  and len(decoded_oovs[0]) > 0:
        for idx, oov in enumerate(decoded_oovs):
            id2word[start_extend_idx + idx] = oov


    return " ".join([id2word[idx] for idx in test_example])


def make_readable(x: str, filter_tok: bool = False, max_w: int = None) -> str:
    '''
    make string readable/clean as preparation for rouge metrics
    :param x: "'' Jenny went to her house ´´ , she eats an <unknown> apple . <eos> <pad> <pad> "
    :param filter_tok: True
    :param max_w: Maximum number of possible windows, for filtering out (multiple) <eos> in dynamic windowing approach
    :return: "Jenny went to her house , she eats an unknown apple ."
    '''
    first_eos_pos = x.find(STOP_DEC)
    if first_eos_pos != -1:  # at least one <eos>
        if max_w:
            eos_pos = tuple(re.finditer(STOP_DEC, x))
            last_eos = min(max_w, len(eos_pos))
            last_eos_pos = eos_pos[last_eos - 1].start()
            x = x[:last_eos_pos].strip()
            x = x.replace(STOP_DEC, "")
        else:
            x = x[:first_eos_pos].strip()

    filter_tokens = []
    if filter_tok:
        filter_tokens += ["''", "``", "´´", PAD]
    # tokens such as "<unknown>" have to be unwrapped to "unknown" otherwise rouge perl script will throw error
    return " ".join([tok for tok in x.replace("<", "").replace(">", "").split(" ") if tok not in filter_tokens])

def printConfig(config):

    print("=======================Config================================")
    for k,v in sorted(config.__dict__.items(), key=lambda x: x[0]):
        if not k.startswith("__") and not callable(v):
            print(k, "|", v)
    print("=============================================================")

def printScoresSelected(rouge_dict, meteor):

    print("ROUGE-1 F1: ", rouge_dict['rouge_1_f_score'])
    print("ROUGE-2 F1: ", rouge_dict['rouge_2_f_score'])
    print("ROUGE-3 F1: ", rouge_dict['rouge_3_f_score'])
    print("ROUGE-L F1: ", rouge_dict['rouge_l_f_score'])
    print("EXACT METEOR: ", meteor)

def encode_oov(oov_list):
    '''
    encodes List[String] into np.array[int]
    :param oov_list: [leifman, self-made, taming]
    :return: [102, 86, 44, ..., 1114112, 98, 53, ..., 1114112, 87, 56, ..., 1114112]
    -> 1114112 = sep_int which separates the encoded words
    '''
    encoded = []
    sep_int = sys.maxunicode + 1
    for oov in oov_list:
        encoded += [ord(c) for c in oov] + [sep_int]

    return np.array(encoded)

def decode_oov(oov_arr):
    '''
    decodes padded np.array[int] into List[String]
    :param oov_arr: [102, 86, 44, ..., 1114112, 98, 53, ..., 1114112, 87, 56, ..., 1114112, 1114112, ..., 1114112]
    :return: ["leifman", "self-made", "taming"]
    '''
    sep_int = sys.maxunicode + 1
    _decode = lambda arr : "".join(chr(token) for token in arr)
    split_by_maxunicode = np.split(oov_arr, np.where(oov_arr == sep_int)[0])
    oov_arr_filtered = [split_by_maxunicode[0]]

    if len(split_by_maxunicode) > 1:
        oov_arr_filtered += [arr[1:] for arr in split_by_maxunicode[1:] if len(arr) > 1]

    return [_decode(arr) for arr in oov_arr_filtered]

def w2v(emb_path, limit=200000):
    word2vec = {}
    with open(emb_path) as f:
        for i, line in enumerate(f):
            if i == 0 and len(line) < 1000:  # skip first line with metadata information
                continue

            word, vec = line.split(' ', 1)
            vec = np.fromstring(vec.strip(), sep=' ')
            word2vec[word] = vec
            if i == limit: break
    return word2vec
# ============================== IN - OUT ===============================================

class MacOSFile(object):
    '''
    File object to encapsulate large objects
    https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    '''
    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))

