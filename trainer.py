# TODO
# - Dump flags & training epoch when suspend / resume
# - 


import theano
import theano.tensor as T
import numpy as np
import gflags

import os
import shutil
import logging
import coloredlogs

import models
import dataproc
from util import *


gflags.DEFINE_integer('n_embed', 20, 'Dimension of word embedding')
gflags.DEFINE_integer('n_hidden', 30, 'Dimension of hidden layer')
gflags.DEFINE_integer('n_context', 25, 'Dimension of hidden layer')
gflags.DEFINE_integer('n_vocab', 502, 'as shown')
gflags.DEFINE_integer('n_layers', 2, 'Number of RNN layers')
gflags.DEFINE_integer('n_doc_batch', 2, 'Documents per batch')
gflags.DEFINE_integer('n_sent_batch', 3, 'Batch size of sentences in a document batch')
gflags.DEFINE_integer('n_epochs', 10, 'Number of epochs')

gflags.DEFINE_float('dropout_prob', 0.5, 'Pr[drop_unit]')

gflags.DEFINE_enum('optimizer', 'RMSProp2', ['AdaDelta', 'AdaGrad', 'RMSProp', 'RMSProp2', 'SGD'],
                   'as shown')
gflags.DEFINE_float('lr', 1e-4, 'Learning rate for all optimizers')
gflags.DEFINE_float('opt_decay', 0.9, 'Decay rate for RMS.+/Ada.+')
gflags.DEFINE_float('opt_momentum', 0.9, 'Momentum for SGD')

gflags.DEFINE_string('dump_prefix', './test', 'as shown')
gflags.DEFINE_string('load_npz', '', 'empty to train from scratch; otherwise resume from corresponding file')
gflags.DEFINE_string('train_data', './data/test.train', 'path of training data')

gflags.DEFINE_bool('test_value', True, 'Compute test value of theano')


flags = None


def train(model, train_batches, valid_batches):

    # TODO: REMOVEME
    model.save('FUCKYOU.npz')

    # == Fix random state ==
    np.random.seed(7297)

    # == Compile theano functions ==
    X_data = T.lmatrix('X_data')
    X_mask = T.fmatrix('X_mask')
    X_pos = T.lmatrix('X_pos')
    X_mask_d = T.fmatrix('X_mask_d')
    Y_data = T.ltensor3('Y_data')
    Y_mask = T.ftensor3('Y_mask')
    Y_mask_d = T.fmatrix('Y_mask_d')

    if flags['test_value']:
        theano.config.compute_test_value = 'warn'
        #
        sum_sent_len = 13
        batch_size = 5
        n_sent = 3
        doc_batch_size = 11
        sent_len = 7
        #
        X_data.tag.test_value = np.zeros((sum_sent_len, batch_size), dtype=np.int64)
        X_mask.tag.test_value = np.ones((sum_sent_len, batch_size), dtype=np.float32)
        X_pos.tag.test_value = np.zeros((n_sent, doc_batch_size), dtype=np.int64)
        X_mask_d.tag.test_value = np.ones((n_sent, doc_batch_size), dtype=np.float32)
        Y_data.tag.test_value = np.zeros((n_sent, sent_len, doc_batch_size), dtype=np.int64)
        Y_mask.tag.test_value = np.ones((n_sent, sent_len, doc_batch_size), dtype=np.float32)
        Y_mask_d.tag.test_value = np.ones((n_sent, doc_batch_size), dtype=np.float32)

    loss, valid_upd, train_upd = model.train([X_data, X_mask, X_pos, X_mask_d], [Y_data, Y_mask, Y_mask_d])

    train_batch = theano.function([X_data, X_mask, X_pos, X_mask_d, Y_data, Y_mask, Y_mask_d],
                                  loss,
                                  updates=train_upd,
                                  on_unused_input='warn', # OPTME
                                  name='train_batch')

    valid_batch = theano.function([X_data, X_mask, X_pos, X_mask_d, Y_data, Y_mask, Y_mask_d],
                                  loss,
                                  updates=valid_upd,
                                  on_unused_input='warn',
                                  name='valid_batch')

    model.init_rng()
  
    # == Train loop ==
    best_model_path = ""
    best_valid_loss = 1e100
    for epoch in xrange(flags['n_epochs']):
        np.random.shuffle(train_batches)
         
        t_loss = []
        v_loss = []
        for batch_id, b in train_batches:
            b_loss = train_batch(*b)
            t_loss.append(b_loss)
            log_info({'type': 'batch', 'id': batch_id, 'loss': float(np.mean(t_loss))})

        for batch_id, b in valid_batches:
            b_loss = valid_batch(*b)
            v_loss.append(b_loss)
            log_info({'type': 'valid', 'id': batch_id, 'loss': float(np.mean(v_loss))})

        valid_loss = float(np.mean(v_loss))
        log_info({'type': 'epoch', 'id': epoch, 
                 'train_loss': float(np.mean(t_loss)), 'valid_loss': valid_loss})
    
        if valid_loss < best_valid_loss:
            log_info({'type': 'epoch_save_new_best', 'id': epoch})
            best_model_path = flags['dump_prefix'] + ('-ep%d.npz' % epoch)
            model.save(best_model_path)

    shutil.copy(best_model_path, flags['dump_prefix'] + '-best.npz')
    log_info({'type': 'training_finished'})
    

def main():
    import sys
    global flags
    sys.argv = gflags.FLAGS(sys.argv)
    flags = gflags.FLAGS.FlagValuesDict()
    models.flags = flags
    model = models.SoftDecoder()
    train_batches, valid_batches = dataproc.load_data(flags)
    train(model, train_batches, valid_batches)


if __name__ == '__main__':
    import coloredlogs
    coloredlogs.install(show_hostname=False, show_name=False)
    logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S')
    main()
