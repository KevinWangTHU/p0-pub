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
                 dropout_prob, dropout_switch, 
                 theano_rng,
                 pref, pdict):
        """
        @param dropout_prob: Probability of dropping out
        @param dropout_switch: theano shared variable; value = 1{use_dropout}
        """
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.W = []
        self.U = []
        self.b = []
        self.dropout_prob = dropout_prob
        self.dropout_switch = dropout_switch
        self.rng = theano_rng
        for i in xrange(n_layers):
            self.W.append(init_matrix_u((n_input, n_hidden * 4), pref + '_w%d' % i, pdict))
            self.U.append(init_matrix_u((n_hidden, n_hidden * 4), pref + '_u%d' % i, pdict))
            self.b.append(init_matrix_u((n_hidden * 4, ), pref + '_b%d' % i, pdict))
            n_input = n_hidden
        self.params = self.W + self.U + self.b
    
    def step(self, x_t, xm_t, pre_c, pre_h, *gpu_args):
        """
        @param x_t:    T(n_batch, n_input)
        @param xm_t:   T(n_batch,), 01 vector indicating whether ith sequence has ended
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
            inp = dropout(inp, self.dropout_switch, self.dropout_prob, self.rng)
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
        @return:         (hiddens, updates), where
                         hidden ~ T(len, n_layers, n_batch, n_hidden).
                         NOTE this function should be able to handle concatenated lists,
                         as mask will reset all states (cell & hidden).
        """
        # TODO: Modify all callers
        # TODO: Check if input[-1] == 0 in scan
        n_batch = inputs.shape[1]

        gen0 = lambda: T.alloc(0., self.n_layers, n_batch, self.n_hidden)
        h_0 = h_0 or gen0()

        [cells, hiddens], update = theano.scan(fn=self.step, 
                                               sequences=[dict(input=inputs, taps=delta_t),
                                                          masks],
                                               outputs_info=[gen0(), h_0],
                                               non_sequences=self.get_params() + [self.dropout_switch],
                                               strict=True)
         
        return hiddens, updates

