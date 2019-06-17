import os
import struct
import re
import sys
import torch.nn as nn
from constants import START_DEC, START_SENT, END_SENT, STOP_DEC, PAD, UNKNOWN, DATADUMP_PATH, DATASET2BIN, EMB_PATH
from sklearn.preprocessing import normalize
from Model.windowing import DynamicScheduler
from tensorflow.core.example import example_pb2
from Auxiliary.utils import encode_oov, pickle_dump, pickle_load, w2v
import random
import numpy as np
import torch


class Batch(object):
    "Batch object holding relevant information for training"

    def __init__(self, batch, device):
        self.size = batch.__len__()
        self.device = device
        # NOTE: dec_max_len creation assumes a top-down sorting according to len
        self.dec_max_len = batch[0].dec_max_len
        self.encoder_oovs = [x.encoder_oovs for x in batch]
        self.enc_inputs_oov = self._to_torch(self._collect(batch, 'encoder_pointer_idx'))
        self.dec_inputs = self._to_torch(self._collect(batch, 'decoder_input')[:, :self.dec_max_len])
        self.dec_targets_oov = self._to_torch(self._collect(batch, 'decoder_target_pointer')[:, :self.dec_max_len])

    def __len__(self):
        return self.size

    def _collect(self, batch, attr):
        return np.stack([getattr(ex, attr) for ex in batch])

    def _to_torch(self, x):
        return torch.from_numpy(x).to(self.device)


class BucketIterator(object):
    def __init__(self, Config, batch_size, device, vocab):
        self.Config = Config
        self.batch_size = batch_size
        self.device = device
        self.max_tgt_in_batch = 0

    def __call__(self, data, shuffle=True):
        if shuffle:
            for p in self.batch(data, self.batch_size * 100):
                p_batch = self.batch(
                    sorted(p, key=lambda x: x.dec_max_len, reverse=True),
                    self.batch_size, self.batch_size_fn)
                l_p_batch = list(p_batch)
                for b in random.sample(l_p_batch, len(l_p_batch)):
                    yield Batch(b, self.device)

        else:
            for b in self.batch(data, self.batch_size):
                yield Batch((sorted(b, key=lambda x: x.dec_max_len, reverse=True)), self.device)

    def batch(self, data, batch_size, batch_size_fn=None):
        """Yield elements from data in chunks of batch_size."""
        if batch_size_fn is None:
            def batch_size_fn(next, count):
                return count
        batch, count = [], 0
        for ex in data:
            batch.append(ex)
            count = batch_size_fn(ex, len(batch))
            if count == batch_size:
                yield batch
                batch, count = [], 0
            elif count > batch_size:
                yield batch[:-1]
                batch, count = batch[-1:], batch_size_fn(ex, 1)
        if batch:
            yield batch

    def batch_size_fn(self, next, ex_count):
        "Augment batch with instances and calculate token number"
        if ex_count == 1:
            self.max_tgt_in_batch = 0
        self.max_tgt_in_batch = max(self.max_tgt_in_batch,  next.dec_max_len)
        tgt_elements = ex_count * self.max_tgt_in_batch
        return tgt_elements


class Vocabulary(object):

    def __init__(self, vocab_path, max_size=50000):
        self.word2id = {}
        self.id2word = None

        for word in [UNKNOWN, PAD, START_DEC, STOP_DEC]:
            self.word2id[word] = len(self.word2id)

        with open(vocab_path, 'r') as f:
            for line in f:
                tokens = line.split()
                if len(tokens) == 2:
                    word = tokens[0]
                    if word in [UNKNOWN, PAD, START_DEC, STOP_DEC, START_SENT, END_SENT]:
                        raise Exception(f"Special tokens such as {word} must not be in vocab file")
                    if word in self.word2id:
                        print(f"Token {word} encountered twice")
                        continue
                    count = len(self.word2id)
                    if count >= max_size:
                        break
                    self.word2id[word] = count

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, wordOrId):
        if isinstance(wordOrId, int):
            if wordOrId not in self.id2word:
                raise ValueError(f"Id {wordOrId} not found in vocab of size {self.__len__()}")
            return self.id2word[wordOrId]
        else:
            return self.word2id[wordOrId] if wordOrId in self.word2id else self.word2id[UNKNOWN]

    def __len__(self):
        return len(self.word2id)


class Instance(object):
    def __init__(self, article, abstract, vocab, Config, word2vec):
        self.abstract = abstract
        self.article = article
        self.Config = Config
        self.vocab = vocab

        self.start_dec_id = vocab[START_DEC]
        self.stop_dec_id = vocab[STOP_DEC]
        self.pad_id = vocab[PAD]
        self.unknown_id = vocab[UNKNOWN]

        # ENCODER - PREPARATION
        # lookup article idx and truncate to enc_max_len
        self.encoder_idx = [vocab[word] for word in article.split()[:self.Config.enc_max_len]]
        self.encoder_idx = np.array(self.encoder_idx)

        self.encoder_pointer_idx = None
        self.encoder_oovs = None

        # add out-of-vocabulary temporary ids to encoder sequence
        self.encoder_pointer_idx = self._add_encoder_oovs(self.encoder_idx)
        # pad shorter documents with PAD token at beginning

        self.encoder_idx = np.pad(self.encoder_idx, (0, self.Config.enc_max_len - len(self.encoder_idx)),
                                  "constant", constant_values=(self.pad_id,))
        self.encoder_pointer_idx = np.pad(self.encoder_pointer_idx, (0, self.Config.enc_max_len - len(self.encoder_pointer_idx)),
                                          "constant", constant_values=(self.pad_id,))
        '''
        else:
            self.encoder_idx = np.pad(self.encoder_idx, (self.Config.enc_max_len-len(self.encoder_idx),0),
                                      "constant", constant_values=(self.pad_id,))
            self.encoder_pointer_idx = np.pad(self.encoder_pointer_idx, (self.Config.enc_max_len-len(self.encoder_pointer_idx),0),
                                              "constant", constant_values=(self.pad_id,))
        '''
        if abstract:
            # DECODER - PREPARATION
            self.decoder_idx = [vocab[word] for word in abstract.split()[:self.Config.dec_max_len]]
            self.decoder_idx = np.array(self.decoder_idx)

            self.decoder_pointer_idx = None
            self.decoder_target_pointer = None

            # add out-of-vocabulary temporary ids from encoding stage to decoder sequence
            self.decoder_pointer_idx = self._add_decoder_oovs(self.decoder_idx)
            self.decoder_target_pointer = np.concatenate((self.decoder_pointer_idx, np.array([self.stop_dec_id]))) \
                                if self.decoder_pointer_idx.shape[0] < self.Config.dec_max_len \
                                else self.decoder_pointer_idx

            self.decoder_input = np.concatenate((np.array([self.start_dec_id]), self.decoder_idx))[:self.Config.dec_max_len]
            self.decoder_target = np.concatenate((self.decoder_idx, np.array([self.stop_dec_id]))) \
                                    if self.decoder_idx.shape[0] < self.Config.dec_max_len \
                                    else self.decoder_idx

            # pad shorter documents with PAD token at the end
            _end_pad = lambda x: np.pad(x, (0, self.Config.dec_max_len - len(x)), "constant", constant_values=(self.pad_id,))
            self.decoder_idx = _end_pad(self.decoder_idx)
            self.decoder_pointer_idx = _end_pad(self.decoder_pointer_idx)
            self.decoder_input = _end_pad(self.decoder_input)
            self.decoder_target = _end_pad(self.decoder_target)
            self.decoder_target_pointer = _end_pad(self.decoder_target_pointer)

            self.dec_max_len = (self.decoder_target_pointer != self.vocab[PAD]).sum()

            if self.Config.windowing and self.Config.w_type == 'dynamic':
                ds = DynamicScheduler(self.vocab, word2vec)
                self.dec_max_len, self.decoder_input, self.decoder_target_pointer = \
                    ds.inst_up(self, self.Config.ws, self.Config.ss, self.Config.enc_max_len, self.Config.dec_max_len)

    def _add_encoder_oovs(self, encoder_idx):
        '''Replace out-of-vocabulary unknown ids with temporary ids for pointer reference'''
        idx = encoder_idx.copy()
        unknown_idx = np.where(self.unknown_id==idx)[0]
        article_list = np.array(self.article.split())[:self.Config.enc_max_len]
        unknown_tokens = article_list[unknown_idx]
        unique_unknown_tokens = sorted(list(set(unknown_tokens)))
        unique_oov_idx = np.arange(len(unique_unknown_tokens)) + self.vocab.__len__()
        oov_idx = np.array([unique_oov_idx[unique_unknown_tokens.index(u)] for u in unknown_tokens])
        idx[unknown_idx] = oov_idx
        self.encoder_oovs = np.array(unique_unknown_tokens)

        return idx

    def _add_decoder_oovs(self, decoder_idx):
        '''Replace out-of-vocabulary ids from encoding stage with their corresponding temporary id
           -> exclusive decoding-oovs will persist with unknown id'''
        idx = decoder_idx.copy()
        abstract_list = np.array(self.abstract.split())[:self.Config.dec_max_len]
        for i, oov in enumerate(self.encoder_oovs):
            oov_idx = np.where(abstract_list == oov)[0]
            idx[oov_idx] = i + self.vocab.__len__()

        return idx

class Generator(object):
    def __init__(self, Config, vocab, device):
        self.Config = Config
        self.vocab = vocab
        self.device = device
        self.bin_partitions = {'train': os.path.join(DATASET2BIN[Config.dataset], 'train.bin'),
                           'val': os.path.join(DATASET2BIN[Config.dataset], 'val.bin'),
                           'test': os.path.join(DATASET2BIN[Config.dataset], 'test.bin')}
        self.instances = None
        self.buckit = BucketIterator(self.Config, self.Config.batch_size * self.Config.dec_max_len, self.device, self.vocab)
        self.word2vec = None


    def __call__(self, partition='train'):
        if not self.instances:
            dumpPath = self._dumpPath(partition)
            if os.path.exists(dumpPath):
                self.instances = pickle_load(dumpPath)
            else:
                if self.Config.windowing and self.Config.w_type == 'dynamic':
                    self.word2vec = w2v(EMB_PATH)
                self.instances = self.instance_list(partition)
        return self.buckit(self.instances, self.Config.shuffle)

    def instance_list(self, partition='train'):
        reader = open(self.bin_partitions[partition], 'rb')
        instances, count = [], 0
        while True:
            len_bytes = reader.read(8)
            if not len_bytes: break
            str_len = struct.unpack('q', len_bytes)[0]
            instance_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]  # get string representation of tensorflow example
            tf_example = example_pb2.Example.FromString(instance_str)  # cast back to tensorflow example
            try:
                abstract = tf_example.features.feature["abstract"].bytes_list.value[0].decode()
                article = tf_example.features.feature["article"].bytes_list.value[0].decode()

            except:
                continue

            abstract = re.sub(f"{START_SENT}", " ", abstract)  # remove start sentence token in abstract
            abstract = re.sub(f"{END_SENT}", ".", abstract)  # replace end sentence token with "."
            abstract = re.sub(r"\. \.", ".", abstract)
            abstract = re.sub(r"\s+", " ", abstract).strip()  # truncate multiple whitespaces and strip
            if len(article) != 0 and len(abstract) != 0:

                inst = Instance(article, abstract, self.vocab, self.Config, self.word2vec)
                instances.append(inst)
                count += 1
                log_border = 500 \
                    if self.Config.windowing and self.Config.w_type == 'dynamic' \
                    else 10000
                if count % log_border == 0:
                    print(f"Processed {count} instances")

        if self.Config.dump:
            pickle_dump(instances, self._dumpPath(partition))

        return instances

    def _dumpPath(self, partition):
        dyn = True \
            if self.Config.windowing and self.Config.w_type == 'dynamic' \
            else False
        outname = f"data_{self.Config.dataset}_{partition}_{self.vocab.__len__()}_{self.Config.enc_max_len}_" + \
                  f"{self.Config.dec_max_len}_{dyn}.pickle"
        return os.path.join(DATADUMP_PATH, outname)


def data_generator(data_path):

    reader = open(data_path, 'rb')
    while True:
        len_bytes = reader.read(8)
        if not len_bytes: break
        str_len = struct.unpack('q', len_bytes)[0]
        instance_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]  # get string representation of tensorflow example
        tf_example = example_pb2.Example.FromString(instance_str)  # cast back to tensorflow example
        try:
            abstract = tf_example.features.feature["abstract"].bytes_list.value[0].decode()
            article = tf_example.features.feature["article"].bytes_list.value[0].decode()

        except:
            continue

        abstract = re.sub(f"{START_SENT}", " ", abstract)  # remove start sentence token in abstract
        abstract = re.sub(f"{END_SENT}", ".", abstract)  # replace end sentence token with "."
        abstract = re.sub(r"\. \.", ".", abstract)
        abstract = re.sub(r"\s+", " ", abstract).strip()  # truncate multiple whitespaces and strip
        if len(article) != 0 and len(abstract) != 0:
            yield article, abstract

def dump_data(dataPath, vocab, config):
    '''

    :param vocab: Vocabulary object
    :param config: {"enc_max_len":400, "dec_max_len":80, "windowing":False, "dataset":"Wiki"}
    :return: - dump data
    '''

    generator = data_generator(dataPath)
    data = []
    count = 0
    for art, abst in generator:
        inst = Instance(abst, art, vocab, config)

        encoded_oovs = encode_oov(inst.encoder_oovs)
        sep_int = sys.maxunicode + 1
        if len(encoded_oovs) <= 500:
            encoded_oovs = np.pad(encoded_oovs, (0, 500-len(encoded_oovs)), "constant", constant_values=(sep_int,))
        else:
            continue  # 9 instances in total have greater length, they are omitted due to unnecessary tensor blowup
        row = np.concatenate((inst.encoder_pointer_idx,
                              inst.decoder_input,
                              inst.decoder_target_pointer,
                              encoded_oovs), axis=0)

        data.append(row[None, :])
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} instances")

    data = np.concatenate(data, axis=0)

    description = "train"
    if dataPath.endswith("val.bin"):
        description = "val"
    elif dataPath.endswith("test.bin"):
        description = "test"

    outname = f"data_{config['dataset']}_{description}_{vocab.__len__()}_{config['enc_max_len']}_{config['dec_max_len']}_{config['windowing']}.pickle"
    pickle_dump(data, os.path.join(DATADUMP_PATH, outname))


def dump_embedding_layer(emb_path, vocab, dataset='CnnDm', dim=300, trainable=True, norm=True):

    # 1) READ EMBEDDINGS FROM FILE AND POPULATE WORD2VEC DICTIONARY
    word2vec = w2v(emb_path)

    # 2) BUILD EMBEDDING MATRIX BY INITIALIZING VOCAB-MATRIX WITH EMBEDDINGS FOUND BY LOOKUP WITH PRETRAINED EMBEDDINGS
    embedding_matrix = np.random.randn(vocab.__len__(), dim)
    no_correspondence = 0
    for i, word in enumerate(list(vocab.word2id.keys())):
        if word in word2vec:
            embedding_matrix[i] = word2vec.get(word)
            continue

        # FALLBACK I - initiliaze OOV collocations "high-profile" with the average of the components "high", "profile"
        if "-" in word:
            num = 0
            vec = np.zeros(dim)

            for component in word.split("-"):
                if component in word2vec:
                    vec += word2vec[component]
                    num += 1

            if num != 0:
                vec /= num
                embedding_matrix[i] = vec

            else:
                no_correspondence += 1

            continue

        # FALLBACK II - initialize numbers with the embedding of "hundred"
        try:
            int_rep = int(word)
            vec = word2vec["hundred"]
            embedding_matrix[i] = vec
            continue

        except ValueError:
            pass

        no_correspondence += 1

    print(f"{no_correspondence} out of {vocab.__len__()} words are initialized randomly")

    if norm:
        embedding_matrix = normalize(embedding_matrix, axis=1, norm='l2')

    # 3) TORCH EMBEDDING LAYER

    N = embedding_matrix.shape[0]
    dim = embedding_matrix.shape[1]
    embedding = torch.from_numpy(embedding_matrix)
    layer = nn.Embedding(N, dim)
    layer.load_state_dict({"weight": embedding})
    layer.weight.requires_grad = trainable

    # 4) DUMP STATE_DICT
    torch.save(layer.state_dict(), os.path.join(os.path.dirname(emb_path), f"dump/initial_{dataset}_{vocab.__len__()}_{dim}.pt"))


if __name__ == "__main__":
    from constants import get_vocab_path
    from Auxiliary.config import Configuration
    import yaml

    with open("conf/train.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    CONFIG = Configuration(**cfg)
    DEVICE = torch.device("cpu")
    vocab = Vocabulary(CONFIG.vocab_path, CONFIG.max_vocab)
    gen = Generator(CONFIG, vocab, DEVICE)
    iter = gen('train')





