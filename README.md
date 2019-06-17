# abs-summ-long-texts

Abstractive Summarization for Long Texts

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Prerequisites/Dependencies

* Python3 (Python3.6 used for testing)
* [Pytorch 1.1.0](https://pytorch.org/)
* [tensorflow](https://www.tensorflow.org/)
* [numpy](https://www.numpy.org/)
* [pyrouge](https://github.com/andersjo/pyrouge) - for evaluation
* [METEOR](https://www.cs.cmu.edu/~alavie/METEOR/) - for evaluation, set environment variable METEOR for jar file meteor-1.5.jar
* [Stanford-Parser](https://stanfordnlp.github.io/CoreNLP/) - for tokenization with stanford-parser.jar


## Hyperparameters
Overview for train.yml. Please not that not all combinations of specifications are possible. Refer to the thesis paper to see which models were trained, i.e. Transformer models do not support pointing and reinforcement learning
### General
| Option        | Example Values | Description  |
| ------------- |:-------------:| -----:|
| dataset       | CnnDm/Wiki/Pubmed | choose dataset |
| dump | True | whether to dump pre-processed dataset for subsequent instant load |
| device        | cuda:0/cpu         | whether to work on cpu/selected gpu |
| num_threads   | 1                 | number of threads used |
| batch_size    | 12                | batch size used for dynamic batching |
| shuffle       | True              | whether to shuffle instances in dataset |
| toy_run       | False             | enables working in debug mode, only 10 batches per epoch |

### Training
| Option        | Example Values | Description  |
| ------------- |:-------------:| -----:|
| criterion     | "torch.nn.NLLLoss() | criterion used for training, negative log likelihood |
| save_interval | 4                 | save model every 4 epochs |
| log_interal | 2 | log example output every 2 epochs |
| val_interval | 4  | validate model every 4 epochs using Rouge metrics |
| num_epochs | 20 | train model for 20 epochs |
| weight decay | 0.00001 | weight decay regularization to counter potential overfitting |
| rl | False | whether to train with mixed training objective (reinforcement learning) |
| bp | 5000 | approximate number of instances for subsequent reinforced training (leave empty otherwise) |
| lambda_ | 0.999 | lambda for weighting in mixed training objective |
| resume | True | resume model from from_save_description and from_epoch |
| from_save_description | XXX | if resume=True, model_XXX will be resumed |
| from_epoch | 12 | if resume=True, model_XXX will be resumed from epoch 12 |
| save_dir | "data/models" | models will be saved under directory |
| save_description | "XXXI" | current model will be saved using description XXXI |
| factor | 1 | factor in learning rate schedule |
| warmup | 4000 | warmup specification in learning rate schedule |
| beta_1 | 0.9 | beta_1 in Adam optimizer |
| beta_2 | 0.999 | beta_2 in Adam optimizer |
| opt_eps | 1e-8 | epsilon in Adam optimizer |

### Embedding 
| Option        | Example Values | Description  |
| ------------- |:-------------:| -----:|
| init | True | whether to initialize embeddings with fastText embeddings |
| emb_name | "data/embeddings/wiki.en.vec" | path to pretrained embeddings file |
| emb_dim | 300 | embedding dimensionality |
| max_vocab | 50000 | vocabulary size |
| norm | True | whether to normalize pretrained embeddings prior to initialization |
| trainable | True | whether to keep adjusting pre-trained embeddings when training end-to-end |

### Sequence-2-Sequence
| Option        | Example Values | Description  |
| ------------- |:-------------:| -----:|
| encoder | "Recurrent"/"Transformer" | specify encoder variant |
| decoder | "Recurrent"/"Transformer" | specify decoder variant |
| enc_max_len | 1160 | maximum number of words/tokens considered when encoding document |
| dec_max_len | 100 | maximum number of words/tokens before summary generation terminates |
| tied_weights | True | whether to apply 3-way-tying of embedding/projection layer |
| eval_beam | True | whether to evaluate using beam search heuristic |
| B | 3 | number of beams |

#### Recurrent Models
| Option        | Example Values | Description  |
| ------------- |:-------------:| -----:|
| num_layers | 1 | number of stacked LSTM/GRU layers |
| hidden_dim | 256 | hidden dimensionality in LSTM/GRU |
| rnn_type | "LSTM"/"GRU" | type of recurrent component |
| attention_type | "Luong"/"Bahdanau" | attention variant |
| tf_prob | 0.8 | probability to switch between teacher forcing and model distribution |
| bias | True | whether to use biases in after matrix multiplication |
| eps | 1e-9 | numeric stabilization in log-calculation |
| pointer | True | whether to use pointer-generator network |

### Transformer 
| Option        | Example Values | Description  |
| ------------- |:-------------:| -----:|
| N | 3 | number of stacks |
| h | 4 | number of heads |
| dropout | 0.1 | - |
| d_model | 256 | model dimensionality |
| d_ff | 1024 | dimensionality of position-wise feedforward network |

### Windowing 
| Option        | Example Values | Description  |
| ------------- |:-------------:| -----:|
| windowing | True | whether to train with windowing |
| w_type | "static"/"dynamic" | windowing variant |
| k | 1.2 | first skewness parameter in "static" windowing |
| d | 0.8 | second skewness parameter in "static" windowing |
| max_corpus_len | 800 | Marjority(\|D\|) - majority document length in corpus |
| ws | 400 | window size |
| ss | 380 | step size |




## Use Cases

The following uses cases demonstrate example usages

### 1) Train 

Configure conf/train.yml with desired parameters and train a model, which might be saved with description "XXX" and stored at epoch 20:

```
python train.py
```

### 2) Eval

Evaluate the model with description "XXX" from epoch 20 on the validation/test set - potentially with METEOR - by specifying conf/eval.yml and call

```
python eval.py
```

### 3) Predict
Specify conf/predict.yml for predicting a summary for a given document which is printed to the console. Set jar_path to the location, where the tokenizer stanford-parser.jar is placed. Use dir_mode=False and put the desired text file - i.e. example.txt - under data/predictions/example.txt. Call

```
python predict.py
```

to generate the summary. Note that for windowing models applied to very long document, the options inf_enc_max_len and inf_dec_max_len need to be specified to ensure extrapolation of the model. I.e.
```
inf_enc_max_len = ws + i * ss; inf_dec_max_len = 100 * i
```
where i must be chosen such that ws + i * ss exceeds the document length.

### 4) Highlight (Visualization Tool)
Specify conf/highlight.yml for visualizing the summary generation for a document - i.e. example.txt - which must be placed under data/predictions/example.txt. Specify inf_enc_max_len/inf_dec_max_len similar to 3) for windowing models on long documents. Once conf/hightlight.yml is specified, open Highlight.ipynb and click through the visualization.

## Authors

* **Leon Schüller** 

## Acknowledgments

Hat tip to the follwing repositories for code/inspiration
* http://nlp.seas.harvard.edu/2018/04/03/attention.html
* https://github.com/abisee/pointer-generator
* https://github.com/ChenRocks/fast_abs_rl
* https://github.com/alesee/abstractive-text-summarization
