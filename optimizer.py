import numpy
import theano
import theano.tensor as T
import math
import collections

# params: symbolic / shared variables
# grads: s.t. params := params + grads
# updates: theano update dictionary

# lr, momentum: packed theano variable 
def SGD(params, grads, lr, updates, flags):
    momentum = flags['opt_momentum']
    grad_m = [theano.shared(numpy.zeros(param.get_value(borrow=True).shape, 
                                        dtype=theano.config.floatX)) \
              for param in params]
    for v, g, gm in zip(params, grads, grad_m):
        n_gm = momentum * gm + (1 - momentum) * g
        updates[gm] = n_gm
        updates[v] = v + lr * n_gm
    return updates

def AdaGrad(params, grads, lr0, updates, flags):
    padding = flags['opt_padding'] if 'opt_padding' in flags else 1e-4
    sq_grads = [theano.shared(numpy.zeros(param.get_value(borrow=True).shape, 
                                        dtype=theano.config.floatX)) \
                for param in params]
    for value, grad, sq_grad in zip(params, grads, sq_grads):
        n_sq_grad = sq_grad + T.sqr(grad)
        updates[sq_grad] = n_sq_grad
        updates[value] = value + lr0 * grad / T.sqrt(n_sq_grad + padding)
    return updates

def RMSProp(params, grads, lr0, updates, flags):
    decay = flags['opt_decay']
    padding = flags['opt_padding'] if 'opt_padding' in flags else 1e-5
    sq_grads = [theano.shared(numpy.zeros(param.get_value(borrow=True).shape, 
                                          dtype=theano.config.floatX)) \
                for param in params]
    for value, grad, sq_grad in zip(params, grads, sq_grads):
        n_sq_grad = decay * sq_grad + (1 - decay) * T.sqr(grad)
        updates[sq_grad] = n_sq_grad
        updates[value] = value + lr0 * grad / T.sqrt(n_sq_grad + padding)
    return updates

def RMSProp2(params, grads, lr, updates, flags):
    decay = flags['opt_decay']
    pad = flags['opt_padding'] if 'opt_padding' in flags else 1e-5
    m_grads = [theano.shared(numpy.zeros(param.get_value(borrow=True).shape, 
                                        dtype=theano.config.floatX)) \
              for param in params]
    m_grad2s = [theano.shared(numpy.zeros(param.get_value(borrow=True).shape, 
                                         dtype=theano.config.floatX)) \
               for param in params]

    for value, grad, m_grad, m_grad2 in zip(params, grads, m_grads, m_grad2s):
        updates[m_grad] = decay * m_grad + (1 - decay) * grad
        updates[m_grad2] = decay * m_grad2 + (1 - decay) * T.sqr(grad)
        updates[value] = value + lr * grad / T.sqrt(m_grad2 - T.sqr(m_grad) + pad)    
    return updates


def AdaDelta(params, grads, lr, updates, flags):
    decay = flags['opt_decay']
    pad = flags['opt_padding'] if 'opt_padding' in flags else 1e-5
    m_grad2s = [theano.shared(numpy.zeros(param.get_value(borrow=True).shape, 
                                         dtype=theano.config.floatX)) \
               for param in params]
    m_deltas = [theano.shared(numpy.zeros(param.get_value(borrow=True).shape, 
                                         dtype=theano.config.floatX)) \
               for param in params]
    for val, g, m_g2, m_delta in zip(params, grads, m_grad2s, m_deltas):
        m_g2_n = decay * m_g2 + (1 - decay) * T.sqr(g)
        delta = lr * T.sqrt(m_delta + pad) / T.sqrt(m_g2_n + pad) * g
        updates[val] = val + delta
        updates[m_g2] = m_g2_n
        updates[m_delta] = decay * m_delta + (1 - decay) * T.sqr(delta)
    return updates


def optimize(params, updates, flags):
    """
    NOTE: grads_loc should be updated with gradients divided by batch_size.
    """
    optimizers = {'SGD': SGD,
                  'AdaGrad': AdaGrad,
                  'RMSProp': RMSProp,
                  'RMSProp2': RMSProp2,
                  'AdaDelta': AdaDelta}
    if updates == {}: # Updates should always be OrderedDict
        updates = collections.OrderedDict()
    grads_loc = [theano.shared(p.get_value() * numpy.asarray(0.0, dtype='float32'),
                               name=p.name+'_grad') \
                 for p in params]
    if flags['clip_grads']:
        grad_norm = T.sqrt(sum([T.sum(g ** 2) for g in grads_loc]))
        scaling = theano.ifelse.ifelse(
                T.lt(grad_norm, flags['max_grad_norm']), 
                T.cast(1, 'float32'),
                T.cast(flags['max_grad_norm'], 'float32') / grad_norm)
        grads = [scaling * g for g in grads_loc]
    else:
        grads = grads_loc
    lr = T.fscalar('opt_lr')
    return lr, grads_loc, optimizers[flags['optimizer']](params, grads, lr, updates, flags)

