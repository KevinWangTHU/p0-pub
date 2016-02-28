import theano
import theano.ifelse
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from optimizer import optimize
import lmdb

from util import *


class LSTM:

    def get_params(self):
        return self.params

    def __init__(self,
                 n_layers, n_input, n_hidden, 
                 dropout,
                 pref, pdict):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.W = []
        self.U = []
        self.b = []
        self.dropout = dropout
        for i in xrange(n_layers):
            self.W.append(init_matrix_u((n_input, n_hidden * 4), pref + '_w%d' % i, pdict))
            self.U.append(init_matrix_u((n_hidden, n_hidden * 4), pref + '_u%d' % i, pdict))
            self.b.append(init_matrix_u((n_hidden * 4, ), pref + '_b%d' % i, pdict))
            n_input = n_hidden
        self.params = self.W + self.U + self.b
    
    def step(self, x_t, xm_t, dropout_mask, pre_c, pre_h, *gpu_args):
        """
        @param x_t:    T(n_batch, n_input)
        @param xm_t:   T(n_batch,), 01 vector indicating whether ith sequence has ended
        @param dropout_mask:
                       T(n_layers, n_batch, max(n_hidden, n_input))
        @param pre_c:  T(n_layers, n_batch, n_hidden).
        @param pre_h:  T(n_layers, n_batch, n_hidden)
        @param W:      T(n_layers, n_input, 4 * n_hidden)
        @param U:      T(n_layers, n_hidden, 4 * n_hidden)
        @param b:      T(n_layers, 4 * n_hidden,)
        @return:       pre_c', pre_h'
        return 0 when beyond EOS
        """
        def slice_(t, i, l=1):
            return t[:, self.n_hidden*i: self.n_hidden*(i+l)]

        cs, hs = [], []

        for l in xrange(self.n_layers):
            inp = x_t if l == 0 else hs[l-1]
            inp = self.dropout(inp, dropout_mask[l, :, 0:inp.shape[1]])
            pre_activation = T.dot(inp, self.W[l]) + T.dot(pre_h[l], self.U[l]) + self.b[l]
            o = T.nnet.sigmoid(slice_(pre_activation, 0))
            f = T.nnet.sigmoid(slice_(pre_activation, 1))
            i = T.nnet.sigmoid(slice_(pre_activation, 2))
            c_tilde = T.tanh(slice_(pre_activation, 3))
            c = T.shape_padright(xm_t) * i * c_tilde + f * pre_c[l]
            h = T.shape_padright(xm_t) * o * T.tanh(c)
            cs.append(c)
            hs.append(h)

        return T.stacklists(cs), T.stacklists(hs)

    def forward(self, inputs, masks, h_0=None, delta_t=0):
        """
        @param inputs:   T((len, n_batch, n_input))
        @param masks:    T((len, n_batch)), 01 matrix
        @param h_0:      h_0. None for all-0.
        @param delta_t:  tap value of inputs for scan. That of mask is not altered.
        @return:         (hiddens, updates, last_hidden), where
                         hidden ~ T(len, n_layers, n_batch, n_hidden),
                         last_hidden is the last non-zero hidden block 
                         NOTE this function should be able to handle concatenated lists,
                         as mask will reset all states (cell & hidden).
        """
        n_len = inputs.shape[0]
        n_batch = inputs.shape[1]
        n_input = inputs.shape[2]

        gen0 = lambda: T.alloc(0., self.n_layers, n_batch, self.n_hidden)
        h_0 = h_0 or gen0()

        dropout_masks = self.dropout.prep_mask((n_len, self.n_layers, n_batch, T.maximum(self.n_hidden, n_input)))

        assert -1 <= delta_t <= 0
        if delta_t == -1:
            inputs = T.concatenate(
                [T.shape_padleft(T.zeros_like(inputs[0])),
                 inputs[:-1]],
                axis=0)

        [cells, hiddens], updates = theano.scan(
                fn = self.step, 
                sequences = [inputs, masks, dropout_masks],
                outputs_info = [gen0(), h_0],
                non_sequences = self.get_params() + [self.dropout.switch],
                strict = True)

        def filter_last(h_t, mask_t, res_tm1):
            mask_t = mask_t.dimshuffle(('x', 0, 'x'))
            return res_tm1 * (1.0 - mask_t) + h_t * mask_t

        last_hidden, _ = theano.foldl(
                fn = filter_last,
                sequences = [hiddens, masks],
                outputs_info = [T.alloc(0., hiddens.shape[1], hiddens.shape[2], hiddens.shape[3])])
         
        return hiddens, updates, last_hidden

