import os
import subprocess
import torch.utils.data
from data import Generator, Instance
from constants import get_vocab_path, PREDICTION_PATH
from Auxiliary.utils import ids2sentence, mask_oov, printConfig, make_readable
from Auxiliary.stepper import Stepper
import yaml

# Configuration
with open("conf/predict.yml", 'r') as ymlfile:
    CFG = yaml.load(ymlfile)

MAP_PATH = os.path.join(PREDICTION_PATH, "mapping.txt")
PARSER_JAR_PATH = CFG['jar_path']

DEVICE = None
if CFG['device'] is None:
    CFG['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(CFG['device'])
state = Stepper.load(CFG['from_save_description'], CFG['from_epoch'], CFG['device'])

CONFIG = state['Config']
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

if CFG['dir_mode']:
    SOURCE_DIR = os.path.join(PREDICTION_PATH, CFG['input']+'/source')
    TOK_DIR = os.path.join(PREDICTION_PATH, CFG['input']+'/tokenized')
    PREDICTION_DIR = os.path.join(PREDICTION_PATH, CFG['input']+'/prediction')
    assert os.path.isdir(SOURCE_DIR), 'Provide source directory'
    if not os.path.isdir(TOK_DIR): os.makedirs(TOK_DIR)
    if not os.path.isdir(PREDICTION_DIR): os.makedirs(PREDICTION_DIR)
    SOURCE_TEXTS = sorted(os.listdir(SOURCE_DIR))

else:
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

if __name__ == "__main__":

    with open(MAP_PATH, "a") as mf:
        if CFG['dir_mode']:
            for src in SOURCE_TEXTS:
                tok = f"tokenized_{src[-7:]}"
                src_path, tok_path = os.path.join(SOURCE_DIR, src), os.path.join(TOK_DIR, tok)
                mf.write(f"{src_path} \t {tok_path}\n")
        else:
            mf.write(f"{PATH} \t {TOK_PATH}\n")


    command = ['java', '-cp', PARSER_JAR_PATH,
               'edu.stanford.nlp.process.PTBTokenizer', '-preserveLines', '-ioFileList', MAP_PATH]
    subprocess.call(" ".join(command), shell=True)

    to_predict = []
    if CFG['dir_mode']:
        for tok in sorted(os.listdir(TOK_DIR)):
            tok_path = os.path.join(TOK_DIR, tok)
            to_predict.append(tok_path)
    else:
        to_predict.append(TOK_PATH)

    for tok in to_predict:
        article = read_text_file(tok)
        instance = Instance(" ".join(article), None, STEPPER.vocab, CONFIG, None)
        print("Article: ", " ".join(article))

        oovs = [instance.encoder_oovs]
        idx = torch.from_numpy(instance.encoder_pointer_idx).unsqueeze(0)
        idx_no_oov = mask_oov(idx, STEPPER.vocab)

        if CONFIG.encoder == 'Recurrent':
            enc_outputs, enc_state = STEPPER.encoder(idx_no_oov)
            dec_first_state = STEPPER.encoder.hidden_final(enc_state)
        else:  # Transformer
            enc_outputs = STEPPER.encoder(idx_no_oov)
            dec_first_state = STEPPER.encoder.hidden_final(enc_outputs)

        STEPPER.bsdecoder.batch_size = 1
        STEPPER.bsdecoder.dec_max_len = CONFIG.dec_max_len
        beam_dec_outputs = STEPPER.bsdecoder(enc_outputs, dec_first_state, idx)
        beam_pred = ids2sentence(beam_dec_outputs[0][0].cpu().detach().numpy(), STEPPER.vocab,
                                 oovs[0])

        print("Prediction", beam_pred)
        if CFG['dir_mode']:
            pred_path = os.path.join(PREDICTION_DIR, f"prediction_{tok[-7:]}")

            with open(pred_path, "w") as pf:
                pf.write(make_readable(beam_pred))

    os.remove(MAP_PATH)

