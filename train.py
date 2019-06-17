import torch
import torch.utils.data
import yaml
from data import Generator, Instance
from Auxiliary.utils import printConfig
from Auxiliary.config import Configuration
from Auxiliary.stepper import Stepper
import time
import logging


class Trainer:
    def __init__(self):
        self.Config = None
        self.device = None
        self.stepper = None
        self.train_iter = None
        self.val_iter = None
        self.init()

    def init(self):

        with open("conf/train.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        self.Config = Configuration(**cfg)

        if self.Config.device is None:
            self.Config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.Config.device)
        #if self.device == torch.device("cuda:0"):
        #    Config.pin_memory = True
        torch.set_num_threads(self.Config.num_threads)

        self.stepper = Stepper(self.Config, self.device)
        self.t_gen = Generator(self.Config, self.stepper.vocab, self.device)
        self.v_gen = Generator(self.Config, self.stepper.vocab, self.device)
        self.train_iter = self.t_gen('train')
        self.val_iter = self.v_gen('val')

    def epoch(self, epoch=None):
        sum_loss = 0
        epoch_time = -time.time()
        forward_time = 0
        backward_time = 0
        optimizer_time = 0

        test_batch, test_dec_outputs, test_beam_dict = None, None, None

        for i, batch in enumerate(self.train_iter):

            forward_time -= time.time()
            if self.Config.rl:
                loss = self.stepper.forward_mixed(batch)
            else:
                _, _, loss, _ = self.stepper.forward_mle(batch)
            sum_loss += loss.item()
            forward_time += time.time()

            # backward pass and optimize
            backward_time -= time.time()
            self.stepper.backward(loss)
            backward_time += time.time()
            optimizer_time -= time.time()
            self.stepper.opt_step()
            optimizer_time += time.time()

            # make test prediction
            if i == 0:
                test_dec_outputs, test_beam_dict, _, _ = self.stepper.predict(batch)
                test_batch = batch

            if self.Config.toy_run and i == 10:  # toy_run stops after 10 batches per epoch
                break

            if self.Config.bp and i*self.Config.batch_size > self.Config.bp:
                break

        epoch_time += time.time()

        # print sum of loss of all batches
        print("epoch={} avg_loss={:.2f} forward={:.3f}s backward={:.3f}s opt={:.3f}s other={:.3f}s total={:.3f}s".format(
                epoch, sum_loss / i, forward_time, backward_time, optimizer_time,
                epoch_time - forward_time - backward_time - optimizer_time, epoch_time))

        return test_batch, test_dec_outputs, test_beam_dict

    def train(self, num_epochs):
        printConfig(self.Config)
        print("Training...")
        self.stepper.train()

        for epoch in range(num_epochs):

            test_batch, test_dec_outputs, test_beam_dict = self.epoch(epoch)  # for test prediction
            self.train_iter = self.t_gen('train')

            if self.Config.save_interval and epoch % self.Config.save_interval == 0:
                self.stepper.save(epoch)

            if self.Config.log_interval and epoch % self.Config.log_interval == 0:
                print("====================TRAIN TOY OUTPUT====================")
                self.stepper.toy_output(test_batch, test_dec_outputs, test_beam_dict)
                print("========================================================")

            if self.Config.val_interval and epoch % self.Config.val_interval == 0:
                print("=====================VALIDATION============================")
                self.eval_validation()
                self.val_iter = self.v_gen('val')  # reset iterator
                print("===========================================================")

            print()

    def eval_validation(self):
        self.stepper.eval()  # eval mode implies no teacher forcing
        sum_loss = 0
        validation_time = -time.time()
        rouge_time, beam_time = 0, 0
        i = 0
        count, beam_count = [], []
        meteor, beam_meteor = None, None
        total_val_count = 0

        rouge_val_dicts, rouge_beam_val_dicts = [], []

        for batch in self.val_iter:
            n = batch.__len__()
            total_val_count += n

            dec_outputs, beam_dict, loss, _ = self.stepper.forward_mle(batch)
            sum_loss += loss.item()
            if beam_dict:
                beam_time += beam_dict['beam_batch_time']

            if i == 0:
                print("====================VAL TOY OUTPUT====================")
                self.stepper.toy_output(batch, dec_outputs, beam_dict)
                print("======================================================")

            i += 1

            # evaluate rouge metrics per batch
            rouge_batch_time = -time.time()
            metric_dict, beam_metric_dict = self.stepper.metric_scores(batch, dec_outputs, beam_dict)
            rouge_batch_time += time.time()
            rouge_time += rouge_batch_time

            if metric_dict:
                rouge_val_dicts.append(metric_dict['rouge_batch_dict'])
                count.append(metric_dict['count'])
                if metric_dict['meteor']:
                    meteor += metric_dict['meteor']

            if beam_metric_dict:
                rouge_beam_val_dicts.append(beam_metric_dict['rouge_batch_dict'])
                beam_count.append(beam_metric_dict['count'])
                if beam_metric_dict['meteor']:
                    beam_meteor += beam_metric_dict['meteor']

            if self.Config.toy_run and i == 10:  # toy-run stops after 10 batches per epoch
                break

        meteor = meteor / count if meteor else meteor
        beam_meteor = beam_meteor / beam_count if beam_meteor else beam_meteor
        self.stepper.metric_output(rouge_val_dicts, rouge_beam_val_dicts, count, beam_count, meteor, beam_meteor)

        validation_time += time.time()
        print("Validation on {} documents: avg_loss={:.2f} rouge_ratio={:.2f} rouge_beam_ratio={:.2f} rouge={:.3f}s beam={:.3f}s total={:.3f}s"
              .format(total_val_count, sum_loss/i, sum(count)/total_val_count, sum(beam_count)/total_val_count, rouge_time, beam_time, validation_time))
        self.stepper.train()


if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.DEBUG)


    trainer = Trainer()
    trainer.train(trainer.Config.num_epochs)