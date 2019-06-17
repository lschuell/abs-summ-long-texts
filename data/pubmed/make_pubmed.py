#Adapted from
#https://github.com/abisee/cnn-dailymail/blob/master/make_datafiles.py
#to process Pubmed dataset in a similar way to CNN/Dailymail

import struct
import pandas as pd
from tensorflow.core.example import example_pb2

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

TRAIN, VAL, TEST = "train.txt", "val.txt", "test.txt"

for partition in [TRAIN, VAL, TEST]:
    df = pd.read_json(partition, lines=True, chunksize=20000)
    data = pd.DataFrame()
    for df_chunk in df:
        data = pd.concat([data, df_chunk])

    bin_path = partition[:-3] + 'bin'

    with open(bin_path, 'wb') as f:
        for index, row in data.iterrows():
            article, abstract = row['article_text'], row['abstract_text']

            article = ' '.join(article)
            abstract = ' '.join(abstract).lower()

            tf_example = example_pb2.Example()
            # return tf_example, article
            tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)

            f.write(struct.pack('q', str_len))
            f.write(struct.pack('%ds' % str_len, tf_example_str))






