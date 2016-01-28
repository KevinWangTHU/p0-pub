import logging
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

    def __init__(self, n_input, n_hidden, pref, pdict):
        self.n_hidden = n_hidden
        self.W = init_matrix_u((n_input, n_hidden * 4), pref + '_w', pdict)
        self.U = init_matrix_u((n_hidden, n_hidden * 4), pref + '_u', pdict)
        self.b = init_matrix_u((n_hidden * 4, ), pref + '_b', pdict)
        # self.h_0 = init_matrix_u((n_hidden, ), pref + '_h_0', pdict)
        self.params = [self.W, self.U, self.b]# , self.h_0]
    
    @staticmethod
    def step_(x_t, xm_t, pre_c, pre_h, W, U, b, n_hidden):
        """
        @param x_t:    T(n_batch, n_input)
        @param xm_t:   T(n_batch,), 01 vector indicating whether ith sequence has ended
        @param pre_c:  T(n_batch, n_input)
        @param pre_h:  T(n_batch, n_input)
        @param W, U:   T(n_input, 4 * n_hidden)
        @param hidden: T(4 * n_hidden,)
        return 0 when beyond EOS
        """

        def slice_(t, i, l=1):
            return t[:, n_hidden*i: n_hidden*(i+l)]
    
        pre_activation = T.dot(x_t, W) + T.dot(pre_h, U) + b
        o = T.nnet.sigmoid(slice_(pre_activation, 0))
        f = T.nnet.sigmoid(slice_(pre_activation, 1))
        i = T.nnet.sigmoid(slice_(pre_activation, 2))
        c_tilde = T.tanh(slice_(pre_activation, 3))
        c = T.shape_padright(xm_t) * i * c_tilde + f * pre_c
        h = T.shape_padright(xm_t) * o * T.tanh(c)
        return c, h
    
    def step(self, x_t, xm_t, pre_c, pre_h):
        return LSTM.step_(x_t, xm_t, pre_c, pre_h, self.W, self.U, self.b, self.n_hidden)

    def forward(self, inputs, masks):
        """
        @param inputs: [T((len, n_batch, n_input))]
        @param masks:  [T((len, n_batch))], 01 matrix
        @return:       [T((len, n_batch, n_hidden))], list of hidden layers
                       Note this function should be able to handle concatenated lists,
                       as mask will reset all states (cell & hidden).
        """
        n_samples = inputs.shape[1]
        zero = lambda: T.alloc(0.0, n_samples, self.n_hidden)
        [cells, hiddens], update = theano.scan(fn=lambda x_t, xm_t, pre_c, pre_h, *args:\
                                                  self.step(x_t, xm_t, pre_c, pre_h),
                                               sequences=[inputs, masks],
                                               outputs_info=[zero(), zero()],
                                               non_sequences=[self.W, self.U, self.b],
                                               strict=True)
        return hiddens


def dropout(data, switch, prob, rng):
    """
    @param switch: 1{use_dropout}
    """
    prob = T.cast(prob, dtype='float32')
    mask = rng.binomial(size=data.shape, n=1, p=prob, dtype='float32')
    return theano.ifelse.ifelse(T.lt(0.1, switch),
                                mask * data,
                                prob * data)


def task_model(flags):
    # Word embedding matrix
    W = init_matrix_u((flags['n_vocab'], flags['n_input']), 'tW')
    # For logistic layer
    U = init_matrix_u((flags['n_hidden'], flags['n_labels']), 'lW')
    b = init_matrix_u((flags['n_labels'],), 'lb')
    lstm = lstm_init(flags['n_input'], flags['n_hidden'])
    params = [W, U, b] + lstm.values()

    # Construct computation graph

    rng = RandomStreams(233)

    x = T.lmatrix('x')                         # (len, n_batch)
    x_mask = T.fmatrix('x_mask')               # (len, n_batch)
    y = T.lvector('y')                         # (n_batch, )
    x_emb = W[x]                               # (len, n_batch, n_input)
    result = lstm_forward(lstm, x_emb, x_mask) # (len, n_b, n_h)
    hidden_mean = T.sum(result, axis=0) / T.shape_padright(T.sum(x_mask, axis=0)) # (n_b, n_h)

    # Dropout
    # Switch for dropout
    dropout_on = theano.shared(np.asarray(1.0, dtype='float32'))
    log_input = dropout(hidden_mean, dropout_on, 1.0 - flags['dropout'], rng)

    # Through logistic layer
    actual = T.nnet.softmax(T.dot(log_input, U) + b)

    # Loss and Grad
    expected = T.extra_ops.to_one_hot(y, flags['n_labels'], dtype='float32')
    loss = -T.mean(T.sum(expected * T.log(actual) + (1 - expected) * T.log(1 - actual), axis=1))
    grad = T.grad(-loss, params)

    # Build function
    updates = optimize(flags['optimizer'], params, grad, {}, flags)
    train_batch = theano.function([x, x_mask, y], 
                                  loss, 
                                  updates=updates,
                                  name='train_batch')

    test_batch = theano.function([x, x_mask, y],
                                 loss,
                                 name='test_batch')

    return train_batch, test_batch, dropout_on


def train_lstm(train, valid, test, flags):
    """
        ...
    """

    np.random.seed(233)

    # Calculate n_labels, n_words
    from itertools import chain
    cmax = lambda arr: max(chain.from_iterable(arr))
    flags['n_labels'] = 1 + max([max(p[1]) for p in [train, valid, test]])
    flags['n_vocab']  = 1 + max([cmax(p[0]) for p in [train, valid, test]])
    
    print flags

    # Divide data into batches & swap dimension
    batch_size_ = flags['batch_size']
    processed = []
    for dat in train, valid, test:
        batches = []
        n_samples = len(dat[0])
        for i in xrange(0, n_samples, batch_size_):
            x, mask, y = lmdb.prepare_data(dat[0][i: i+batch_size_], 
                                           dat[1][i: i+batch_size_], 
                                           maxlen=flags['max_len'])
            if not (x is None):
                batches.append([x, mask, np.array(y)])
        processed.append(batches)
        batch_size_ = flags['valid_batch_size']
    train, valid, test = processed

    logging.info('Remain training data: %s' % len(train))

    # Build model
    train_batch, test_batch, dropout_on = task_model(flags)

    # Training
    for ep in range(flags['n_epoch']):
        # Shuffle batches
        np.random.shuffle(train)

        step = 0
        ep_train_loss = []
        ep_length = []
        dropout_on.set_value(1.0)
        for x_b, xm_b, y_b in train:
            loss = train_batch(x_b, xm_b, y_b)
            ep_train_loss += [loss]
            ep_length.append(x_b.shape[0])
            step += 1
            if step % flags['report_every'] == 1:
                logging.info('\tep %d: mean train error %.3f magic %d' % (ep, np.mean(ep_train_loss), np.mean(ep_length)))

        # Evaluate on validation and test set
        dropout_on.set_value(0.0)
        metric = []
        for dat in [valid, test]:
            cur_loss = 0
            for x_b, xm_b, y_b in dat: 
                cur_loss += test_batch(x_b, xm_b, y_b)
            metric.append(cur_loss / len(dat)) 

        ep_train_loss = np.mean(ep_train_loss)
        logging.info("epoch %d: average loss = %.4f; vl = %.4f, tl = %.4f" % \
                     (ep, ep_train_loss, metric[0], metric[1]))


def main():
    import coloredlogs
    coloredlogs.install(show_hostname=False, show_name=False)
    logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S')

    flags = {'n_input': 128,
             'n_hidden': 128,
             'optimizer': 'RMSProp2',
             'batch_size': 16,
             'valid_batch_size': 64,
             'lr': 1e-4,
             'opt_padding': 1e-6,
             'opt_decay': 0.9,
             'l2_decay': 0.00,
             'n_epoch': 100,
             'max_len': 100,
             'report_every': 16,
             'dropout': 0.5,
            }

    train, valid, test = lmdb.load_data()
    train_lstm(train, valid, test,
               flags)

if __name__ == '__main__':
    main()
