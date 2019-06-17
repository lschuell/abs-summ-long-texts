import os
import subprocess
import torch.utils.data
from data import Instance
import re
import numpy as np
from constants import get_vocab_path, PREDICTION_PATH, STOP_DEC
from Auxiliary.utils import ids2sentence, mask_oov, printConfig, make_readable
from Auxiliary.stepper import Stepper
import yaml
from IPython.display import HTML

# Configuration
with open("conf/highlight.yml", 'r') as ymlfile:
    CFG = yaml.load(ymlfile)

MAP_PATH = os.path.join(PREDICTION_PATH, "mapping.txt")
PARSER_JAR_PATH = CFG['jar_path']

DEVICE = None
if CFG['device'] is None:
    CFG['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(CFG['device'])
state = Stepper.load(CFG['from_save_description'], CFG['from_epoch'], CFG['device'])
CONFIG = state['Config']
assert(CONFIG.decoder == 'Recurrent'), 'Highlighting only for Recurrent Decoder'
if CFG['inf_enc_max_len'] and CFG['inf_dec_max_len'] and CONFIG.windowing and CONFIG.w_type == "dynamic":
    CONFIG.enc_max_len = eval(CFG['inf_enc_max_len'])
    CONFIG.dec_max_len = eval(CFG['inf_dec_max_len'])
CONFIG.device = CFG['device']
CONFIG.resume = True
CONFIG.from_save_description = CFG['from_save_description']
CONFIG.from_epoch = CFG['from_epoch']
CONFIG.vocab_path = get_vocab_path(CONFIG.dataset)
CONFIG.eval_beam = True
CONFIG.B = CFG['B']
printConfig(CONFIG)
STEPPER = Stepper(CONFIG, DEVICE)
STEPPER.eval()

PATH = os.path.join(PREDICTION_PATH, CFG['input'])
TOK_PATH = os.path.join(PREDICTION_PATH, "tok_" + CFG['input'])


def read_text_file(text_file):
    '''Read text from file'''
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            if line.strip() != '':
                lines.append(line.strip())
    return lines


def vis_dict():

    with open(MAP_PATH, "a") as mf:
        mf.write(f"{PATH} \t {TOK_PATH}\n")

    command = ['java', '-cp', PARSER_JAR_PATH,
               'edu.stanford.nlp.process.PTBTokenizer', '-preserveLines', '-ioFileList', MAP_PATH]
    subprocess.call(" ".join(command), shell=True)

    article = read_text_file(TOK_PATH)
    article = " ".join(article)
    instance = Instance(article, None, STEPPER.vocab, CONFIG, None)

    idx = torch.from_numpy(instance.encoder_pointer_idx).unsqueeze(0)
    idx_no_oov = mask_oov(idx, STEPPER.vocab)

    if CONFIG.encoder == 'Recurrent':
        enc_outputs, enc_state = STEPPER.encoder(idx_no_oov)
        dec_first_state = STEPPER.encoder.hidden_final(enc_state)
    else:  # Transformer
        enc_outputs = STEPPER.encoder(idx_no_oov)
        dec_first_state = STEPPER.encoder.hidden_final(enc_outputs)

    STEPPER.decoder.dec_max_len = CONFIG.dec_max_len
    dec_outputs, att_weights = STEPPER.decoder(enc_outputs, dec_first_state, None, idx)
    pred = torch.argmax(dec_outputs.transpose(1, 2), dim=-1).squeeze().cpu().numpy()
    pred = ids2sentence(pred, STEPPER.vocab, instance.encoder_oovs)

    if CONFIG.windowing and CONFIG.w_type == 'dynamic':
        num_w = STEPPER.decoder.windower.scheduler.num_w(instance.encoder_pointer_idx, CONFIG.ws, CONFIG.ss)
        if pred.find(STOP_DEC) != -1:
            eos_pos = tuple(re.finditer(STOP_DEC, pred))
            last_eos = min(num_w, len(eos_pos))
            last_eos_pos = eos_pos[last_eos - 1].start()
            pred = pred[:last_eos_pos].strip()
            pred = pred.replace(STOP_DEC, "-->")
    else:
        pred = make_readable(pred, False)

    transitions = None
    if CONFIG.windowing:
        if CONFIG.w_type == 'static':
            transitions = STEPPER.decoder.windower(instance.encoder_pointer_idx)
        else: #dynamic
            transitions = np.where(np.array(pred.split(" ")) == "-->")[0] + 1

    slen_ = pred.split(" ").__len__()
    alen_ = article.split(" ").__len__()

    w_d_ = {"weights": att_weights.squeeze().detach().cpu().numpy()[:slen_, :alen_], "summary": pred.split(" "),
               "article": article.split(" "), "transitions": transitions}

    os.remove(MAP_PATH)

    return w_d_


def summary_div(ex, snapshot=True, hue=270):
    rad = 0
    if CONFIG.windowing:
        a_len, s_len = len(ex['article']), len(ex['summary'])
        remainder = min(a_len, CONFIG.enc_max_len) - CONFIG.ws
        rest = 1 if remainder % CONFIG.ss == 0 else 2
        num_w = remainder // CONFIG.ss + rest if remainder > 0 else 1
        rad = 360 // num_w

    transitions = list(ex['transitions']) if type(ex['transitions']) == np.ndarray else None

    if snapshot:
        curr_str = ""
        for j, str_ in enumerate(ex['summary']):
            if transitions and j == transitions[0]:
                _ = transitions.pop(0)
                hue += rad
            curr_str += hprint(str_, f"hsl({hue}, 100%, 85%)") + " "

        return '''<div class="Summaries">''' + curr_str + '''</div>\n'''

    else:
        sum_str = []
        for i in range(len(ex['weights'])):
            if transitions and i == transitions[0]:
                _ = transitions.pop(0)
                hue += rad
            curr_str = ""
            for j, str_ in enumerate(ex['summary']):
                if j > i: break
                if i == j:
                    curr_str += '<span style="text-decoration:underline; text-decoration-color: hsl(' + str(
                        hue) + ', 100%, 50%)">' + str_ + '</span> '
                    # curr_str += hprint(str_, f"hsl(270, 100%, 50%)") + " "
                else:
                    curr_str += "<span>" + str_ + "</span> "
            sum_str.append(curr_str)

        out_str = '''<div class="Summaries">'''
        for i, str_ in enumerate(sum_str, 1):
            dp = "block" if i == 1 else "none"
            out_str += '''<div class="Summary''' + str(
                i) + '''" style="display: ''' + dp + '''"> ''' + str_ + '''</div>'''
        out_str += '''</div>'''
        return out_str


def article_div(ex, snapshot=True, hue=270):
    transitions = None
    rad = 0
    eff_len = ex['weights'].shape[1]
    if CONFIG.windowing:
        a_len, s_len = len(ex['article']), len(ex['summary'])
        remainder = min(a_len, CONFIG.enc_max_len) - CONFIG.ws
        rest = 1 if remainder % CONFIG.ss == 0 else 2
        num_w = remainder // CONFIG.ss + rest if remainder > 0 else 1
        rad = 360 // num_w
        transitions = ex['transitions']
        eff_trans = list(transitions[transitions < s_len])
        eff_len = min(a_len, len(eff_trans) * CONFIG.ss + CONFIG.ws)
        eff_trans = [0] + eff_trans + [s_len]
        weights = np.zeros((s_len, eff_len))

        for i in range(len(eff_trans) - 1):
            start_sum = eff_trans[i]
            end_sum = eff_trans[i + 1]
            start_art = i * CONFIG.ss
            end_art = eff_len if i == (len(eff_trans) - 2) \
                else i * CONFIG.ss + CONFIG.ws
            weights[start_sum:end_sum, start_art:end_art] = ex['weights'][start_sum:end_sum, 0:(end_art - start_art)]
        transitions = list(transitions)
    else:
        weights = ex['weights']

    if snapshot:  # condense into maximal attention weights
        bc = np.bincount(weights.argmax(1), weights.max(1))
        cl = np.clip(bc, a_min=0, a_max=1)
        weights = np.pad(cl, (0, weights.shape[1] - len(cl)), "constant", constant_values=(0,))
        bp = CONFIG.ss
        adjust_b = False

        mask = weights == 0
        color_int = np.round(97 - weights * 27).astype(int)
        color_int[mask] = 100
        curr_str = ""
        for i, str_ in enumerate(ex['article'][:eff_len]):
            if i == bp:
                hue += rad // 2
                bp = bp + (CONFIG.ws - CONFIG.ss) if adjust_b == False else bp + CONFIG.ss
                adjust_b = not adjust_b

            curr_str += hprint(str_, f"hsl({hue}, 100%, {color_int[i]}%)") + " "
        return '''<div class="Articles">''' + curr_str + '''</div>\n'''
    else:
        art_str = []
        for k, w_t in enumerate(weights):
            if transitions and k == transitions[0]:
                _ = transitions.pop(0)
                hue += rad

            mask = w_t == 0
            color_int = np.round(97 - w_t * 47).astype(int)
            color_int[mask] = 100
            curr_str = ""
            for i, str_ in enumerate(ex['article'][:eff_len]):
                curr_str += hprint(str_, f"hsl({hue}, 100%, {color_int[i]}%)") + " "
            art_str.append(curr_str)
        out_str = '''<div class="Articles">'''
        for i, str_ in enumerate(art_str, 1):
            dp = "block" if i == 1 else "none"
            out_str += '''<div class="Article''' + str(
                i) + '''" style="display: ''' + dp + '''"> ''' + str_ + '''</div>\n'''
        out_str += '''</div>\n'''
        return out_str


def hprint(text, col="hsl(180, 100%, 95%)"):
    return '<span style="background-color: '+col+'">'+text+'</span>'


def highlight_vis(ex, snapshot=True, hue=270):
    ART_DIV = article_div(ex, snapshot, hue)
    SUM_DIV = summary_div(ex, snapshot, hue)
    s_len = len(ex['summary'])

    html_str = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <title>Model Attention Visualization</title>
    <style type="text/css">
        button {
            padding: 5px 10px;
            font-size: 14px;
        }
        .ProBar {
          margin-top: 4px;
          width: 100%;
          background-color: hsl(270, 20%, 80%);
        }
        #Bar {
          width: ''' + str(1 / s_len * 100) + '''%;
          height: 30px;
          background-color: hsl(270, 20%, 20%);
        }
        .Header {
          color: hsl(270, 20%, 20%);
          font-weight: bold;
          font-style: italic;
        }
        .HeadWrapper {
          border-style: solid;
          border-color: hsl(270, 20%, 20%);
          padding: 5px;
        }
    </style>
    <script src="https://code.jquery.com/jquery-1.12.4.min.js"></script>
    <script type="text/javascript">
        var curr_id = 1;
        var pbar = document.getElementById("Bar"); 
        var max_id = ''' + str(s_len) + ''';
        $(document).ready(function(){
            $(".Next").click(function(){
                $(".Article" + curr_id).css('display', 'none');
                $(".Summary" + curr_id++).css('display', 'none');
                $(".Article" + curr_id).css('display', 'block');
                $(".Summary" + curr_id).css('display', 'block');
                pbar.style.width = (curr_id/max_id)*100 + '%';
            });
        });
        $(document).ready(function(){
            $(".Previous").click(function(){
                $(".Article" + curr_id).css('display', 'none');
                $(".Summary" + curr_id--).css('display', 'none');
                $(".Article" + curr_id).css('display', 'block');
                $(".Summary" + curr_id).css('display', 'block');
                pbar.style.width = (curr_id/max_id)*100 + '%';
            });
        });
    </script>
    </head>
    <body>
        <div class="TopBar">
        <button class="Next" type="button">Next</button>
        <button class="Previous" type="button">Previous</button>
        <div class="ProBar"><div id="Bar"></div></div>
        <div class="HeadWrapper"><div class="Header"> ARTICLE: </div>''' + ART_DIV + '''</div>
        <div class="HeadWrapper"><div class="Header"> PREDICTED SUMMARY: </div>''' + SUM_DIV + '''</div>
    </body>
    </html>
    '''

    snap_html_str = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <title>Model Attention Visualization</title>
    <style type="text/css">
        .Header {
          color: hsl(270, 20%, 20%);
          font-weight: bold;
          font-style: italic;
        }
        .HeadWrapper {
          border-style: solid;
          border-color: hsl(270, 20%, 20%);
          padding: 5px;
        }
    </style>
    </head>
    <body>
        <div class="HeadWrapper"><div class="Header"> ARTICLE: </div>''' + ART_DIV + '''</div>
        <div class="HeadWrapper"><div class="Header"> PREDICTED SUMMARY: </div>''' + SUM_DIV + '''</div>
    </body>
    </html>
    '''
    return snap_html_str if snapshot else html_str


if __name__ == "__main__":
    w_ = vis_dict()
    str_ = article_div(w_)
    highlight_vis(w_, hue=270)