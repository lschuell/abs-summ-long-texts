import os
import torch.utils.data
from data import Generator
from constants import get_vocab_path
from Model.windowing import StaticScheduler, Windower
from Auxiliary.evaluation import merge_dicts
from Auxiliary.stepper import Stepper
from Auxiliary.utils import printConfig
import time
import yaml

# Configuration
with open("conf/eval.yml", 'r') as ymlfile:
    CFG = yaml.load(ymlfile)

DEVICE = None
if CFG['device'] is None:
    CFG['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(CFG['device'])
state = Stepper.load(CFG['from_save_description'], CFG['from_epoch'], CFG['device'])
CONFIG = state['Config']
CONFIG.device = CFG['device']
CONFIG.resume = True
CONFIG.from_save_description = CFG['from_save_description']
CONFIG.from_epoch = CFG['from_epoch']
CONFIG.vocab_path = get_vocab_path(CONFIG.dataset)
CONFIG.eval_beam = True
CONFIG.meteor = CFG['meteor']
CONFIG.dump = CFG['dump']
CONFIG.batch_size = CFG['batch_size']
printConfig(CONFIG)
STEPPER = Stepper(CONFIG, DEVICE)
STEPPER.eval()

V_GEN = Generator(CONFIG, STEPPER.vocab, STEPPER.device)

def validate():
    validation_time = -time.time()
    sum_loss = 0
    i = 0
    total_val_count, beam_count, beam_meteor = 0, [], 0
    rouge_beam_val_dicts = []
    val_iter = V_GEN(CFG['partition'])
    for batch in val_iter:
        n = batch.__len__()
        total_val_count += n

        _, beam_dict, loss, _ = STEPPER.forward_mle(batch)
        sum_loss += loss.item()

        i += 1
        # evaluate rouge metrics per batch
        metric_dict, beam_metric_dict = STEPPER.metric_scores(batch, None, beam_dict, with_meteor=CONFIG.meteor)

        rouge_beam_val_dicts.append(beam_metric_dict['rouge_batch_dict'])
        beam_count.append(beam_metric_dict['count'])
        if beam_metric_dict['meteor']:
            beam_meteor += beam_metric_dict['count'] * beam_metric_dict['meteor']

        if CFG['toy_run'] and i == 10:  # toy-run stops after 10 batches per epoch
            break

    beam_meteor = beam_meteor / sum(beam_count) if beam_meteor else beam_meteor

    if len(rouge_beam_val_dicts) > 0:
        final_ = merge_dicts(rouge_beam_val_dicts, beam_count)
        validation_time += time.time()
        SAVE_PATH = os.path.join("output/evals", CONFIG.from_save_description)
        exists = os.path.exists(SAVE_PATH)
        with open(SAVE_PATH, 'a') as f:
            if not exists:
                if CONFIG.windowing:
                    if CONFIG.w_type == 'static':
                        f.write("Partition, B, k, d, ws, ss, ROUGE-1, ROUGE-2, ROUGE-3, ROUGE-L, METEOR EXACT, time \n")
                    else:
                        f.write("Partition, B, ws, ss, ROUGE-1, ROUGE-2, ROUGE-3, ROUGE-L, METEOR EXACT, time \n")
                else:
                    f.write("Partition, B, ROUGE-1, ROUGE-2, ROUGE-3, ROUGE-L, METEOR EXACT, time \n")
            f.write(f"{CFG['partition']}, {STEPPER.bsdecoder.B}, ")
            if CONFIG.windowing:
                if CONFIG.w_type == 'static':
                    f.write(f"{STEPPER.decoder.windower.scheduler.k}, {STEPPER.decoder.windower.scheduler.d}, ")
                f.write(f"{CONFIG.ws}, {CONFIG.ss}, ")
            f.write(f"{final_['rouge_1_f_score']}, {final_['rouge_2_f_score']}, ")
            f.write(f"{final_['rouge_3_f_score']}, {final_['rouge_l_f_score']}, {beam_meteor}, {validation_time} \n")

if __name__ == "__main__":

    if CONFIG.windowing and CONFIG.w_type == 'static':
        for k in CFG['k']:
            for d in CFG['d']:
                for B in CFG['B']:
                    scheduler = StaticScheduler(k, d, CONFIG)
                    windower = Windower(scheduler, CONFIG, STEPPER.vocab)
                    STEPPER.bsdecoder.set_windower(windower)
                    STEPPER.bsdecoder.B = B
                    validate()

    else:
        for B in CFG['B']:
            STEPPER.bsdecoder.B = B
            validate()

