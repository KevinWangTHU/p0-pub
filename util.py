import json
import logging
import operator
import theano
import numpy as np


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


def dropout(data, switch, prob, rng):
    """
    @param prob:   Pr[drop_unit]
    @param switch: 1{use_dropout}
    """
    prob = T.cast(1 - prob, dtype='float32')
    mask = rng.binomial(size=data.shape, n=1, p=prob, dtype='float32')
    return theano.ifelse.ifelse(T.lt(0.1, switch),
                                mask * data,
                                prob * data)
