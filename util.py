import theano
import numpy as np

def init_matrix_u(shape, name=''):
    """
    @type shape: tuple
    @return:     theano.SharedVariable
    """
    if len(shape) == 1: 
        m = np.sqrt(6.0 / (shape[0] + 1))
    else:
        m = np.sqrt(6.0 / (shape[0] + shape[1]))
    return theano.shared(np.random.uniform(-m, +m, shape).astype('f'), name=name)

