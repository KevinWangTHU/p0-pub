# Simple RNN encoder-decoder
# Input of this model is a single sentence for a document.
# :

import theano
import theano.tensor as T
import numpy as np

import lstm
from models import WordDecoder
from util import *

class SimpleRNN:
    """
    Naive RNN encoder/decoder model.
    Parameter size = 
        (n_embed + n_hid) * n_hid * 4 * n_layers * 2 +
        (n_embed + n_hid) * n_vocab
    = 300 * 200 * 4 * 2 * 2 + 300 * 100000 = (6*16)e4 + 3e7 ~ 3e7 => 30M (120M mem)
    For this model to work $n_sent_batch == n_doc_batch$ must holds.
    """

    def get_params(self):
        return self.encoder.get_params() + self.decoder.get_params() + [self.embed]

    def __init__(self, flags):
        # Load parameters if required
        if flags['load_npz'].strip() != "":
            pdict = np.load(flags['load_npz'].strip())
            np.random.set_state(pdict['__np_random_state__'])
        else:
            pdict = None

        self.embed = init_matrix_u((flags['n_vocab'], flags['n_embed']), 'embed', pdict)
        # Simple Encoder-Decoder
        self.flags = flags
        self.dropout = Dropout(flags['dropout_prob'])
        self.encoder = lstm.LSTM(
            flags['n_layers'], flags['n_embed'], flags['n_hidden'],
            self.dropout,
            'simplernn_enc', pdict)
        #
        self.decoder = WordDecoder(self.embed, pdict, self.dropout)

    def train(self, X, Y):
        """
        @param X[0..3]: Input doc info. 
                        X[0] ~ T(*max_sent_len, *n_sent_batch) [i64, data]
                        X[1] ~ T(*max_sent_len, *n_sent_batch) [f32, mask]
                        X[2], X[3]: unused
                        `X[0][:, i]` is the (concatenated) sentence of document $i$
        @param Y:       Expected sentences, where
        @param Y[0]:    T(1, *sent_len, *n_doc_batch) [i64]
        @param Y[1]:    Mask of Y[0], shape=Y[0].shape, dtype=f32
        @param Y[2]:    unused
        @return: (loss, upd for validator, upd for trainer)
        """
        hiddens, upd_enc, last_hid = self.encoder.forward(self.embed[X[0]], X[1]) # (len, nl, n_sent_batch, nh)
        (_, prob), upd_dec = self.decoder.decode(last_hid, Y[0][0], Y[1][0])
        loss = -T.sum(prob) / T.cast(X[0].shape[1], 'float32')
        grad = T.grad(-loss, self.get_params())
        rng_updates = concat_updates(upd_enc, upd_dec)
        return self.get_params(), loss, grad, rng_updates

    def save(self, file_):
        # OPTME: duplicacted with models.SoftDecoder.save
        params = self.get_params()
        pdict = {}
        for p in params:
            assert not (p.name in pdict)
            pdict[p.name] = p.get_value()
        pdict['__np_random_state__'] = np.array(np.random.get_state(), dtype='object')
        pdict['__theano_mrg_rstate__'] = np.array(self.dropout.rng.rstate, dtype='object')
        pdict['__theano_mrg_state_updates__'] = np.array(self.dropout.rng.state_updates, dtype='object')
        np.savez(file_, **pdict)

    def init_rng(self):
        """
        Restore theano rng state. Must be called _after_ compilation
        """
        if self.flags['load_npz'].strip() != "":
            self.dropout.rng.rstate = self.pdict['__theano_mrg_rstate__']
            for (su2, su1) in zip(self.dropout.rng.state_updates, self.pdict['__theano_mrg_state_updates__'][0]):
                su2[0].set_value(su1[0].get_value())

    def test(self, X):
        """
        @param X[]: Input document. numpy.ndarray of the same shape as .train
        @return:    Generated highlights as 
                    [[(log-likelihood, [token-list]) for each beam] for each doc]
        """
        if not hasattr(self, 'encode'):
            data = T.lmatrix('data')
            mask = T.fmatrix('mask')
            hid, upd, last_hid = self.encoder.forward(self.embed[data], mask)
            self.encode = theano.function(
                [data, mask],
                last_hid,
                updates=upd)

        max_sent_len = X[0].shape[0]
        n_sent_batch = X[0].shape[1]
        ret = []
        for i in xrange(n_sent_batch):
            hid_i = self.encode(X[0][:, i].reshape((max_sent_len, 1)),
                                X[1][:, i].reshape((max_sent_len, 1)))
            ret_i = self.decoder.search(hid_i)
            ret.append([(b[0], [b[2]]) for b in ret_i])

        return ret

