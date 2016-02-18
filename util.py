# coding=utf-8

import json
import logging
import operator
import theano
import theano.tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

def log_info(msg):
    logging.info(json.dumps(msg).replace(', ', ',\t'))


def init_matrix_u(shape, name='', pdict=None):
    """
    @type shape: tuple
    @return:     theano.SharedVariable
    """
    if pdict:
        assert name in pdict
        log_info({'type': 'data', 'value': 'loaded matrix %s (%s)' % (name, str(pdict[name].shape))})
        return theano.shared(pdict[name])
    if len(shape) == 1: 
        m = np.sqrt(6.0 / (shape[0] + 1))
    else:
        m = np.sqrt(6.0 / (shape[0] + shape[1]))
    return theano.shared(np.random.uniform(-m, +m, shape).astype('f'), name=name)


def concat(l):
    return reduce(operator.add, l)


def copy_random_state(g1, g2):
    if isinstance(g1.rng, MRG_RandomStreams):
        g2.rng.rstate = g1.rng.rstate
    for (su1, su2) in zip(g1.rng.state_updates, g2.rng.state_updates):
        su2[0].set_value(su1[0].get_value())


class Dropout:

    def __init__(self, prob):
        """
        @param prob:   Pr[drop_unit]
        @param switch: 1{use_dropout}
        """
        self.prob = prob
        self.switch = theano.shared(np.array(1.0).astype('f'))
        self.rng = MRG_RandomStreams(seed=7297)

    def __call__(self, data):
        prob = T.cast(1 - self.prob, dtype='float32')
        mask = self.rng.binomial(size=data.shape, n=1, p=prob, dtype='float32')
        return theano.ifelse.ifelse(T.lt(0.1, self.switch),
                                    mask * data,
                                    prob * data)

    # TODO: dump & restore parameters


def concat_updates(upd0, upd1):
    """
    Concatenate consecutive updates.
    @param upd0, upd1: OrderedUpdates.
    #                   computation graphs related to upd0 and upd1 should be 
    #                   dependent.
    @return:           OrderedUpdates
    天下本无事, 庸人自扰之.
    """
    return upd0 + upd1
#    ret = [(key1, theano.clone(upd1[key1], replace=upd0)) for key1 in upd1]
#    return theano.OrderedUpdates(ret)
