#Adapted from
#https://github.com/abisee/cnn-dailymail/blob/master/make_datafiles.py
#to process Wikihow dataset in a similar way to CNN/Dailymail

import os
import shutil
import struct
import subprocess
import collections
import re
from tensorflow.core.example import example_pb2

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PARSER_JAR_PATH = "~/stanford-parser-full-2018-10-17/stanford-parser.jar"
WIKI_DIR = "../wiki"
WIKI_PLAIN_DIR, WIKI_TOK_DIR = os.path.join(WIKI_DIR, "plain"), os.path.join(WIKI_DIR, "tokenized")
WIKI_FINISHED_FILES_DIR = "../wiki_finished_files"

VOCAB_SIZE = 200000

def read_text_file(text_file):
    '''Read text from file'''
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def makedir(Dir):
    if os.path.exists(Dir):
        shutil.rmtree(Dir)
    os.makedirs(Dir)

def clean_string(line):
    '''Clean messy sentence start and ending'''
    clean = re.sub(r"^, ", "", line)
    clean = re.sub(r"\. [;,]$", ".", clean)
    clean = re.sub(r"\.[;,]", ".", clean)
    clean = re.sub(r"(\w{3,})\.(\w{3,})", r"\1 \2", clean)
    return clean

def prepare_tok(title):

    lines = read_text_file(f"articles/{title}.txt")
    lines = [line.lower() for line in lines]
    lines = [clean_string(line) for line in lines]

    article = []
    summary = []
    next_is_summary = False
    for idx, line in enumerate(lines):
        if line == "" or line == ",":
            continue  # empty/irrelevant line
        elif line.startswith("@article"):
            continue
        elif line.startswith("@summary"):
            next_is_summary = True
        elif next_is_summary:
            summary.append(line)
            next_is_summary = False
        else:
            article.append(line)

    plain_path, tok_path = os.path.join(WIKI_PLAIN_DIR, title), os.path.join(WIKI_TOK_DIR, title)
    with open(plain_path, "a") as pf:
        for line in summary: pf.write(f"{line} \n")
        pf.write("@SEP \n")
        for line in article: pf.write(f"{line} \n")

    with open("../mapping.txt", "a") as mf:
        mf.write(f"{plain_path} \t {tok_path}\n")


def write_to_bin(idx):
    '''Write to binary files train.bin, val.bin, test.bin and store a vocabulary object'''
    titles = os.listdir(WIKI_TOK_DIR)
    print(f"{titles.__len__()} articles have been tokenized")
    train_c, val_c, test_c = 0, 0, 0

    vocab_counter = collections.Counter()

    with open(os.path.join(WIKI_FINISHED_FILES_DIR, "train.bin"), 'wb') as w_train, \
            open(os.path.join(WIKI_FINISHED_FILES_DIR, "val.bin"), 'wb') as w_val, \
            open(os.path.join(WIKI_FINISHED_FILES_DIR, "test.bin"), 'wb') as w_test:
        for i, title in enumerate(titles):
            tok_path = os.path.join(WIKI_TOK_DIR, title)
            lines = read_text_file(tok_path)

            article = []
            summary = []
            next_is_article = False
            for line in lines:
                if line == "" or line == ",":
                    continue  # empty/irrelevant line
                elif line.startswith("@SEP"):
                    next_is_article = True
                    continue
                elif next_is_article:
                    article.append(line)
                else:
                    summary.append(line)

            # Make article into a single string
            article = ' '.join(article)

            # Make abstract into a single string, putting <s> and </s> tags around the sentences
            abstract = ' '.join(["%s %s %s" % (SENTENCE_START, line, SENTENCE_END) for line in summary])

            tf_example = example_pb2.Example()
            # return tf_example, article
            tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)

            if title in idx['train']:
                w_train.write(struct.pack('q', str_len))
                w_train.write(struct.pack('%ds' % str_len, tf_example_str))

                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')
                abs_tokens = [t for t in abs_tokens if
                              t not in [SENTENCE_START, SENTENCE_END]]
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]
                tokens = [t for t in tokens if t != ""]
                vocab_counter.update(tokens)
                train_c += 1

            elif title in idx['val']:
                w_val.write(struct.pack('q', str_len))
                w_val.write(struct.pack('%ds' % str_len, tf_example_str))
                val_c += 1

            elif title in idx['test']:
                w_test.write(struct.pack('q', str_len))
                w_test.write(struct.pack('%ds' % str_len, tf_example_str))
                test_c += 1

    print(f"\t {train_c} fall to the train partition")
    print(f"\t {val_c} fall to the val partition")
    print(f"\t {test_c} fall to the test partition")

    # write vocab to file
    with open(os.path.join(WIKI_FINISHED_FILES_DIR, "vocab"), 'w') as writer:
        for word, count in vocab_counter.most_common(VOCAB_SIZE):
            writer.write(word + ' ' + str(count) + '\n')

if __name__ == "__main__":

    TITLES = read_text_file("titles.txt")
    train_idx, val_idx, test_idx = map(read_text_file, ["all_train.txt", "all_val.txt", "all_test.txt"])

    idx = {"train": train_idx, "val":val_idx, "test":test_idx}

    makedir(WIKI_DIR)
    makedir(WIKI_FINISHED_FILES_DIR)
    makedir(WIKI_PLAIN_DIR)
    makedir(WIKI_TOK_DIR)

    error_c = 0
    # prepare for tokenization
    for title in TITLES:
        if os.path.exists(f"articles/{title}.txt"):
            prepare_tok(title)
        else:
            error_c += 1

    print(f"{error_c} of titles in titles.txt cannot be retrieved from articles directory")
    # tokenize
    command = ['java', '-cp', PARSER_JAR_PATH,
               'edu.stanford.nlp.process.PTBTokenizer', '-preserveLines', '-ioFileList', '../mapping.txt']
    subprocess.call(" ".join(command), shell=True)
    write_to_bin(idx)
    shutil.rmtree(WIKI_DIR)






