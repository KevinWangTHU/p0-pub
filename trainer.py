import theano
import theano.tensor as T
import numpy as np
from nltk.translate import bleu_score
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
gflags.DEFINE_bool('pick_model', True, 'Use pick model')
gflags.DEFINE_bool('cur_learning', True, 'Use curriculum learning')
gflags.DEFINE_bool('reverse_input', True, 'Reverse output when predicting')
gflags.DEFINE_bool('lvt', False, 'Apply large-vocabulary trick')
gflags.DEFINE_bool('attend_pos', False, 'Concatenate sentence position info when computing attention probabilities')

gflags.DEFINE_integer('n_embed', 100, 'Dimension of word embedding')
gflags.DEFINE_integer('n_hidden', 200, 'Dimension of hidden layer')
gflags.DEFINE_integer('n_context', 200, 'Dimension of context layer')
gflags.DEFINE_integer('n_vocab', 100000, 'as shown')
gflags.DEFINE_integer('n_out_vocab', 50000, '#{Top k words used in prediction}')
gflags.DEFINE_integer('n_layers', 4, 'Number of RNN layers') # n_layers=1 causes CE in theano 0.7; fixed in dev version
gflags.DEFINE_integer('n_doc_batch', 10, 'Documents per batch')
gflags.DEFINE_integer('n_sent_batch', 20, 'Batch size of sentences in a document batch')
gflags.DEFINE_integer('n_epochs', 10, 'Number of epochs')
gflags.DEFINE_integer('n_beam', 5, 'Number of candidates in beam search')
gflags.DEFINE_integer('n_max_sent', 30, 'Maximum sentence length allowed in beam search')
# gflags.DEFINE_float('w_entropy', 0.05, 'Penalty for att weight entropy. Only used in SentExtractor.') # max-ent ~ 5
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
gflags.DEFINE_string('train_data', './data/100k5', 'path of training data')
gflags.DEFINE_bool('test_value', False, 'Compute test value of theano') # Issue with MRG
gflags.DEFINE_bool('trunc_data', False, 'Use ~100 docs for quick test')


flags = None


def compile_functions_legacy(model):
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
    allowed_words = T.lvector('allowed_words')

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
        allowed_words.tag.test_value = np.ones((17, ), dtype=np.int64)
        model.dropout.switch.set_value(0.)

    if not flags['lvt']:
        allowed_words = None

    params, loss, grad, rng_updates = \
            model.train([X_data, X_mask, X_pos, X_mask_d],
                        [Y_data, Y_mask, Y_mask_d], allowed_words)
    lr, grad_shared, opt_updates = optimizer.optimize(params, {}, flags) 

#    params = model.get_params()
#    loss = sum([T.mean(p) for p in params])
#    grad = T.grad(-loss, params)
#    rng_updates = theano.OrderedUpdates()
#    lr, grad_shared, opt_updates = optimizer.optimize(params, {}, flags) 

    func_params = [X_data, X_mask, X_pos, X_mask_d, Y_data, Y_mask, Y_mask_d]
    if allowed_words:
        func_params.append(allowed_words)

    get_loss = theano.function(func_params,
                               loss,
                               updates = rng_updates + OrderedDict(zip(grad_shared, grad)),
                               on_unused_input = 'warn', # Take care
                               name = 'get_loss')
    update_params = theano.function([lr], [],
                                    updates = opt_updates,
                                    name = 'do_optimize')

    model.init_rng()
    return get_loss, update_params


def compile_functions(model):
    X_data = T.lmatrix('X_data')
    X_mask = T.fmatrix('X_mask')
    X_pos = T.lmatrix('X_pos')
    X_mask_d = T.fmatrix('X_mask_d')
    Y_loss = T.fmatrix("Y_loss")
    Y_pre = T.fmatrix("Y_loss")
    epoch = T.fscalar("epoch")

    params, loss, grad, rng_updates = \
        model.train(epoch,
                    [X_data, X_mask, X_pos, X_mask_d],
                    [Y_loss, Y_pre])
    func_params = [epoch, X_data, X_mask, X_pos, X_mask_d, Y_loss, Y_pre]
    lr, grad_shared, opt_updates = optimizer.optimize(params, {}, flags)

    get_loss = theano.function(func_params,
                               loss,
                               updates = rng_updates + OrderedDict(zip(grad_shared, grad)),
                               on_unused_input = 'warn', # Take care
                               name = 'get_loss')
    update_params = theano.function([lr], [],
                                    updates = opt_updates,
                                    name = 'do_optimize')
    model.init_rng()
    return get_loss, update_params


def test_model(model, test_batches):
    import IPython
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.evaluation.rouge import rouge_n
    from sumy.models.dom import Sentence

    Max_pick = 4
    tokenizer = Tokenizer("english")
    model.dropout.switch.set_value(0.)

    r1 = []
    r2 = []
    count = 0
    for batch_id, b, ori_texts, ori_hlts in test_batches:
        probs = model.test(b[:4])
        probs = probs.T.tolist() # ~ [n_batch, n_len]
        for i, prob in enumerate(probs):
            ori_text = ori_texts[i]
            ori_hlt = ori_hlts[i]
            pick_prob = sorted(zip(prob, range(len(prob))), reverse=True)[:Max_pick]
            pick_idx = [b for a, b in pick_prob]
            pick_sent = [ori_text[idx] for idx in pick_idx]

            pick_sent = [Sentence(sent, tokenizer) for sent in pick_sent]
            ori_hlt = [Sentence(hlt, tokenizer) for hlt in ori_hlt]

            r1.append(rouge_n(pick_sent, ori_hlt, 1))
            r2.append(rouge_n(pick_sent, ori_hlt, 2))
            count += 1
    print count
    print sum(r1)/count
    print sum(r2)/count
    return

def test_model_legacy(model, test_batches):
    import IPython
    bleus = []
    generated_highlights = []
    model.dropout.switch.set_value(0.)
    for batch_id, _, b, data_hlts in test_batches:
        model_hlts = model.test(b[0:4], 5, flags['n_max_sent']) # ~ [[(float(LogP), [[int] * n_hlts])] * n_beam] * n_batch
        IPython.embed()
        model_hlts = [hl[0][1] for hl in model_hlts]  # Remove all but the most probable text
        generated_highlights += model_hlts
        for model_hlt, data_hlt in zip(model_hlts, data_hlts):
            # BLEU for concatenated highlights
            bleu = bleu_score.bleu([concat(data_hlt)], concat(model_hlt), weights=[0.25]*4)
            bleus.append(bleu)
        log_info({'type': 'test_batch', 'bleu': np.mean(bleus)})

    if flags['dump_highlights']:
        with open(flags['train_data'] + '.dict') as fin:
            wdict = cPickle.load(fin)
        words = dict([(i, w) for w, i in wdict.items()])
        generated_highlights = [[[words[i] for i in hlt]
                                 for hlt in doc_hlt]
                                for doc_hlt in generated_highlights]
        with open(flags['dump_prefix'] + ".highlights", "w") as fout:
            cPickle.dump(generated_highlights, fout)


def new_model():
    if flags['simplernn']:
        return simple_model.SimpleRNN(flags)
    elif flags['pick_model']:
        return models.RNNPicker()
    else:
        return models.SoftDecoder()


def train(train_batches, valid_batches):
    # == Fix random state & compile ==
    np.random.seed(7297)

    # == Compile ==
    if flags['compile']:
        model = new_model()
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

    log_info({'type': 'function_ready'})
    # == Train loop ==

    best_model_path = ""
    best_valid_loss = 1e100
    for epoch in xrange(flags['n_epochs']):
        np.random.shuffle(train_batches)

        t_loss = []
        v_loss = []
        dropout_switch.set_value(1.0) # 1{use_dropout}
        # for batch_id, b, _, _ in train_batches:
        for batch_id, b, ori_texts, ori_hlts in train_batches:
            #
            while True: # question
                # Hack for memory shortage. NOTE: The random state may be affected
                try:
                    b_loss = get_loss(epoch, *b)
                except Exception as e:
                    if str(e.args).find('allocat') != -1:
                        log_info({'type': 'batch_MLE', 'id': batch_id, 'value': str(e.args), 'magic': gc.collect() + gc.collect()})
                        continue
                    else:
                        raise e
                break
            #
            n_doc_batch = b[2].shape[1]
            update_params(float(flags['lr'] * n_doc_batch))
            #
            t_loss.append(b_loss)
            log_info({'type': 'batch', 'id': batch_id, 'loss': float(np.mean(t_loss))})

        dropout_switch.set_value(0.0)
        # for batch_id, _, b, _ in valid_batches:
        for batch_id, b, ori_texts, ori_hlts in valid_batches:
            while True:
                try:
                    # b_loss = get_loss(epoch, *b)
                    b_loss = get_loss(flags["n_epochs"], *b)
                except Exception as e:
                    if str(e.args).find('llocation') != -1:
                        log_info({'type': 'valid_MLE', 'id': batch_id, 'value': str(e.args), 'magic': gc.collect() + gc.collect()})
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
            best_valid_loss = valid_loss

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
    if flags['mode'] == 'train':
        train_batches, valid_batches = dataproc.load_data(flags)
        train(train_batches, valid_batches)
    else:
        test_batches = dataproc.load_test_data(flags)
        test_model(new_model(), test_batches)


if __name__ == '__main__':
    main()
