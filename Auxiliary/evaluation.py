from pyrouge.utils import log
import pyrouge
import tempfile
import numpy as np
from typing import List, Dict
from Auxiliary.utils import ids2sentence, make_readable
from cytoolz import curry
import subprocess
import os
import re


def sorted_values(rouge_dict):
    '''
    :param rouge_dict: rouge result dictionary {'rouge_1_f_score':0.95918, 'rouge_1_f_score_cb':0.98918 ...}
    :return: list of values sorted by key [0.95918, 0.98918, ...]
    '''
    values = []
    for k, v in sorted(rouge_dict.items(), key=lambda x: x[0]):
        values.append(v)
    return np.array(values)


def merge_dicts(rouge_dicts: List[Dict], counts: List[int]):
    '''
    :param rouge_dicts: list of rouge dicts [dict1, dict2, ...]
    :return: merged result dict with averaged scores - avg_dict
    '''

    num = len(rouge_dicts)
    values = [sorted_values(rd) for rd in rouge_dicts]
    values = np.stack(values)
    counts = np.transpose(np.array([counts,]*values.shape[-1]))
    values = (values * counts).sum(0)/sum(counts)
    result_dict = rouge_dicts[0]
    pos = 0
    for k, v in sorted(result_dict.items(), key=lambda x: x[0]):
        result_dict[k] = values[pos]
        pos += 1
    return result_dict


def rouge_scores(system_dir, model_dir):
    '''
    calculate rouge scores (dict) between reference/system summaries and model summaries
    :param system_dir:
    :param model_dir:
    :return:
    '''
    r = pyrouge.Rouge155()
    r.system_filename_pattern = 'system.(\d+).txt'
    r.model_filename_pattern = 'model.#ID#.txt'
    r.system_dir = system_dir
    r.model_dir = model_dir
    output = r.convert_and_evaluate()
    return r.output_to_dict(output)


@curry
def read_file_1line(file_dir, file_name):
    with open(os.path.join(file_dir, file_name)) as f:
        return ' '.join(f.read().split())


def meteor_score(system_dir, model_dir):
    '''
    Calculate exact Meteor score
    :param system_dir:
    :param model_dir:
    :return:
    '''
    try:
        _METEOR_PATH = os.environ['METEOR']
    except KeyError:
        print('Warning: METEOR is not configured')
        _METEOR_PATH = None

    assert _METEOR_PATH is not None

    system_filename_pattern = re.compile(r'system.(\d+).txt')
    model_filename_pattern = re.compile(r'model.(\d+).txt')
    system_summaries = sorted([sys for sys in os.listdir(system_dir) if system_filename_pattern.match(sys)],
                              key=lambda x: int(x.split('.')[1]))
    model_summaries = sorted([mod for mod in os.listdir(model_dir) if model_filename_pattern.match(mod)],
                             key=lambda x: int(x.split('.')[1]))

    with open(os.path.join(system_dir, 'system.txt'), 'w') as system_f, \
            open(os.path.join(model_dir, 'model.txt'), 'w') as model_f:
        system_f.write('\n'.join(map(read_file_1line(system_dir), system_summaries)) + '\n')
        model_f.write('\n'.join(map(read_file_1line(model_dir), model_summaries)) + '\n')
    cmd = f"java -Xmx2G -jar {_METEOR_PATH} {os.path.join(model_dir, 'model.txt')} {os.path.join(system_dir, 'system.txt')} -l en -norm"
    result = subprocess.check_output(cmd.split(' '))
    final_score = re.search('Final score:\s+(\d\.\d+)$', result.decode()).group(1)

    return float(final_score)


def setup_and_eval(system_summaries, model_summaries, with_meteor=False):
    '''
    temporarily setup rouge structure and evaluate given summaries
    :param system_summaries: list of abstract-summaries in text ["John Stein went ...", ...]
    :param model_summaries: list of corresponding model-summaries ["John walked to ...", ...]
    :return: count of evaluated pairs, rouge result dictionary
    '''
    system_dir_name = "system_summaries"
    model_dir_name = "model_summaries"
    #log.get_global_console_logger().setLevel(logging.WARNING)
    log.get_global_console_logger().disabled = True
    sents_filter = lambda summary: [sent.split(" ").__len__() > 1 for sent in summary.split(".")]
    summary_filter = lambda summary, sents_filter: np.array(summary.split("."))[sents_filter]

    with tempfile.TemporaryDirectory() as tmp_dir:
        system_dir = os.path.join(tmp_dir, system_dir_name)
        model_dir = os.path.join(tmp_dir, model_dir_name)
        os.mkdir(system_dir)
        os.mkdir(model_dir)
        count = 0

        for sys, mod in zip(system_summaries, model_summaries):
            system_sents_filter = sents_filter(sys)
            model_sents_filter = sents_filter(mod)
            system_summary_filter = summary_filter(sys, system_sents_filter)
            model_summary_filter = summary_filter(mod, model_sents_filter)

            if system_summary_filter.__len__() > 0 and model_summary_filter.__len__() > 0:
                system_file = f"system.{count}.txt"
                model_file = f"model.{count}.txt"

                with open(os.path.join(system_dir, system_file), "w") as sf:
                    for i, sent in enumerate(system_summary_filter, 1):
                        sf.write(sent.lstrip() + "\n") if i < len(system_summary_filter) \
                            else sf.write(sent.lstrip())

                with open(os.path.join(model_dir, model_file), "w") as mf:
                    for i, sent in enumerate(model_summary_filter, 1):
                        mf.write(sent.lstrip() + "\n") if i < len(model_summary_filter) \
                            else mf.write(sent.lstrip())

                count += 1

        output = rouge_scores(system_dir, model_dir)
        meteor = None
        if with_meteor:
            meteor = meteor_score(system_dir, model_dir)

    log.get_global_console_logger().disabled = False

    return count, output, meteor

def pred2scores(batch, system_summaries, preds, vocab, num_w, batch_wise=True, with_meteor=False):

    model_summaries = [ids2sentence(pred, vocab, oov) for pred, oov in zip(preds, batch.encoder_oovs)]
    model_summaries = [make_readable(s, True, num_w[i]) for i, s in enumerate(model_summaries)]

    if batch_wise:
        try:
            count, rouge_batch_dict, meteor = setup_and_eval(system_summaries, model_summaries, with_meteor=with_meteor)
            out = {'count': count, 'rouge_batch_dict': rouge_batch_dict, 'meteor': meteor}
        except:
            out = None
    else:
        out = []
        for sys_sum, mod_sum in zip(system_summaries, model_summaries):
            try:
                _, rouge_dict, _ = setup_and_eval([sys_sum], [mod_sum])
                out.append(rouge_dict['rouge_l_f_score'])
            except:
                out.append(0)
    return out