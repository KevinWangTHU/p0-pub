# TODO
# - Test deploy

import theano
import theano.tensor as T
import numpy as np
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
import dataproc
from util import *


gflags.DEFINE_integer('n_embed', 100, 'Dimension of word embedding')
gflags.DEFINE_integer('n_hidden', 200, 'Dimension of hidden layer')
gflags.DEFINE_integer('n_context', 200, 'Dimension of context layer')
gflags.DEFINE_integer('n_vocab', 100000, 'as shown')
gflags.DEFINE_integer('n_layers', 1, 'Number of RNN layers')
gflags.DEFINE_integer('n_doc_batch', 10, 'Documents per batch')
gflags.DEFINE_integer('n_sent_batch', 20, 'Batch size of sentences in a document batch')
gflags.DEFINE_integer('n_epochs', 10, 'Number of epochs')

gflags.DEFINE_float('dropout_prob', 0.5, 'Pr[drop_unit]')

gflags.DEFINE_enum('optimizer', 'RMSProp2', ['AdaDelta', 'AdaGrad', 'RMSProp', 'RMSProp2', 'SGD'],
                   'as shown')
gflags.DEFINE_float('lr', 1e-5, 'Learning rate for all optimizers')
gflags.DEFINE_float('opt_decay', 0.9, 'Decay rate for RMS.+/Ada.+')
gflags.DEFINE_float('opt_momentum', 0.9, 'Momentum for SGD')

gflags.DEFINE_bool('compile', True, 'Recompile theano functions')
gflags.DEFINE_bool('func_output', False, 'Dump function if compiled')
gflags.DEFINE_string('func_input', '', 'Use compiled function if valid')
gflags.DEFINE_string('dump_prefix', '-', 'as shown; - for autocreate')
gflags.DEFINE_string('load_npz', '', 'empty to train from scratch; otherwise resume from corresponding file')
gflags.DEFINE_string('train_data', './data/100k.train', 'path of training data')

gflags.DEFINE_bool('test_value', False, 'Compute test value of theano') # Issue with MRG


flags = None


def compile_functions(model):
    """
    @return: compiled theano functions (train_batch, valid_batch)
    """
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

    params, loss, grad, rng_updates = \
            model.train([X_data, X_mask, X_pos, X_mask_d], [Y_data, Y_mask, Y_mask_d])
    grad_shared, opt_updates = optimizer.optimize(params, {}, flags) 

#    params = model.get_params()
#    loss = sum([T.mean(p) for p in params])
#    grad = T.grad(-loss, params)
#    rng_updates = theano.OrderedUpdates()
#    grad_shared, opt_updates = optimizer.optimize(params, {}, flags) 

    get_loss = theano.function([X_data, X_mask, X_pos, X_mask_d, Y_data, Y_mask, Y_mask_d],
                               loss,
                               updates = rng_updates + OrderedDict(zip(grad_shared, grad)),
                               on_unused_input = 'warn',
                               name = 'get_loss')

    update_params = theano.function([], [],
                                    updates = opt_updates,
                                    name = 'do_optimize')

    model.init_rng()

    return get_loss, update_params


def train(train_batches, valid_batches):
    # == Fix random state & compile ==
    np.random.seed(7297)

    # == Compile ==
    if flags['compile']: 
        model = models.SoftDecoder()
        dropout_switch = model.dropout.switch
        get_loss, update_params = compile_functions(model)
        if flags['func_output']:
            out_path = flags['dump_prefix'] + "_fn.pickle"
            sys.setrecursionlimit(1048576)
            with open(out_path, 'wb') as fout:
                cPickle.dump((dropout_switch, get_loss, update_params), fout, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        with open(flags['func_input'], 'rb') as fin:
            dropout_switch, get_loss, update_params = cPickle.load(fin)
    
    # == Train loop ==

    best_model_path = ""
    best_valid_loss = 1e100
    for epoch in xrange(flags['n_epochs']):
        np.random.shuffle(train_batches)
         
        t_loss = []
        v_loss = []
        dropout_switch.set_value(1.0) # 1{use_dropout}
        for batch_id, b in train_batches:
            # Hack for memory leak (?)
            # FIXME: The random state may be affected
            while True: 
                try:
                    b_loss = get_loss(*b)
                except Exception as e:
                    if str(e.args).find('allocat') != -1:
                        log_info({'type': 'batch_MLE', 'id': batch_id, 'value': str(e.args), 'magic': gc.collect() + gc.collect()})
                        continue
                    else:
                        raise e
                break
            update_params()
            t_loss.append(b_loss)
            log_info({'type': 'batch', 'id': batch_id, 'loss': float(np.mean(t_loss))})

        dropout_switch.set_value(0.0) 
        for batch_id, b in valid_batches:
            while True:
                try:
                    b_loss = get_loss(*b)
                except Exception as e:
                    if str(e.args).find('llocation') != -1:
                        log_info({'type': 'valid_MLE', 'id': batch_id, 'value': str(e.args)})
                        continue
                    else:
                        raise e
                break
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
    if flags['dump_prefix'] == '-':
        import os
        import datetime
        import re
        now = str(datetime.datetime.today())
        now = re.sub('[^\w-]', '-', now)
        os.system("mkdir -p ./dump/%s/" % now)
        flags['dump_prefix'] = "./dump/%s/f" % now

    log_setup()
    log_info(flags)
    #
    train_batches, valid_batches = dataproc.load_data(flags)
    train(train_batches, valid_batches)


if __name__ == '__main__':
    main()
