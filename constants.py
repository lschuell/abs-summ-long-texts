import os

PUBMED_DIR = os.path.join(os.getcwd(), "data/pubmed")
WIKI_DIR = os.path.join(os.getcwd(), "data/wiki_finished_files")
CNNDM_DIR = os.path.join(os.getcwd(), "data/cnn_dm_finished_files")
EMB_PATH = os.path.join(os.getcwd(), "data/embeddings/wiki.en.vec")
MODELDUMP_PATH = os.path.join(os.getcwd(), "data/models")
DATADUMP_PATH = os.path.join(os.getcwd(), "data/dump")
PREDICTION_PATH = os.path.join(os.getcwd(), "data/predictions")
DATASET2BIN = {'CnnDm' : os.path.join(os.getcwd(), 'data/cnn_dm_finished_files'),
                'Wiki' : os.path.join(os.getcwd(), 'data/wiki_finished_files'),
                'Pubmed': os.path.join(os.getcwd(), 'data/pubmed')
                }

UNKNOWN = "<unknown>"
PAD = "<pad>"
START_DEC = "<sos>"
STOP_DEC = "<eos>"
START_SENT = "<s>"
END_SENT = "</s>"
DOT = "."

def get_vocab_path(dataset):
    if dataset == "Wiki": vocab_path = os.path.join(WIKI_DIR, "vocab")
    elif dataset == "Pubmed": vocab_path = os.path.join(PUBMED_DIR, "vocab")
    else: vocab_path = os.path.join(CNNDM_DIR, "vocab")
    return vocab_path