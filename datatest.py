import gflags

import gc
import os
import shutil
import sys
import logging
import coloredlogs
import cPickle
from collections import OrderedDict

import optimizer
import models
import simple_model
import dataproc
from util import *


gflags.DEFINE_enum('mode', 'train', ['train', 'test'], 'as shown')
gflags.DEFINE_bool('__ae__', False, 'Train an autoencoder')
gflags.DEFINE_bool('dump_highlights', True, 'dump generated highlights in test mode')
gflags.DEFINE_bool('simplernn', False, 'Use SimpleRNN')
gflags.DEFINE_bool('reverse_input', True, 'Reverse output when predicting')
gflags.DEFINE_bool('lvt', False, 'Apply large-vocabulary trick')
gflags.DEFINE_bool('attend_pos', True, 'Concatenate sentence position info when computing attention probabilities')

gflags.DEFINE_integer('n_embed', 100, 'Dimension of word embedding')
gflags.DEFINE_integer('n_hidden', 200, 'Dimension of hidden layer')
gflags.DEFINE_integer('n_context', 200, 'Dimension of context layer')
gflags.DEFINE_integer('n_vocab', 100000, 'as shown')
gflags.DEFINE_integer('n_out_vocab', 50000, '#{Top k words used in prediction}')
gflags.DEFINE_integer('n_layers', 1, 'Number of RNN layers') # n_layers=1 causes CE in theano 0.7; fixed in dev version
gflags.DEFINE_integer('n_doc_batch', 10, 'Documents per batch')
gflags.DEFINE_integer('n_sent_batch', 20, 'Batch size of sentences in a document batch')
gflags.DEFINE_integer('n_epochs', 10, 'Number of epochs')
gflags.DEFINE_integer('n_beam', 5, 'Number of candidates in beam search')
gflags.DEFINE_integer('n_max_sent', 30, 'Maximum sentence length allowed in beam search')
gflags.DEFINE_float('dropout_prob', 0.2, 'Pr[drop_unit]')
gflags.DEFINE_bool('clip_grads', True, 'Clip gradients')
gflags.DEFINE_float('max_grad_norm', 5, 'Maximum gradient norm allowed (divided by batch_size)')

gflags.DEFINE_enum('optimizer', 'SGD', ['AdaDelta', 'AdaGrad', 'RMSProp', 'RMSProp2', 'SGD'],
                   'as shown')
gflags.DEFINE_float('lr', 0.1, 'Learning rate for all optimizers')
gflags.DEFINE_float('opt_decay', 0.9, 'Decay rate for RMS.+/Ada.+')
gflags.DEFINE_float('opt_momentum', 0, 'Momentum for SGD')

gflags.DEFINE_bool('compile', True, 'Recompile theano functions')
gflags.DEFINE_bool('func_output', False, 'Dump function if compiled')
gflags.DEFINE_string('func_input', '', 'Use compiled function if valid')
gflags.DEFINE_string('dump_prefix', '-', 'as shown; - for autocreate')
gflags.DEFINE_string('load_npz', '', 'empty to train from scratch; otherwise resume from corresponding file')
gflags.DEFINE_string('wordvec', None, 'Pre-trained word vector file. Check data/proc for format.')
gflags.DEFINE_string('train_data', './data/100ktest', 'path of training data')
gflags.DEFINE_bool('test_value', False, 'Compute test value of theano') # Issue with MRG
gflags.DEFINE_bool('trunc_data', False, 'Use ~100 docs for quick test')


def log_setup():
    import coloredlogs
    coloredlogs.install(show_hostname=False, show_name=False)
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
    datefmt = '%a, %d %b %Y %H:%M:%S'
    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        datefmt=datefmt)
    fh = logging.FileHandler(flags['dump_prefix'] + '.log')
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    logging.getLogger().addHandler(fh)


def main():
    global flags
    sys.argv = gflags.FLAGS(sys.argv)
    flags = gflags.FLAGS.FlagValuesDict()
    models.flags = flags

    log_setup()
    log_info(flags)

    train_batches, valid_batches = dataproc.load_data(flags)
    import IPython
    IPython.embed()


if __name__ == "__main__":
    main()
