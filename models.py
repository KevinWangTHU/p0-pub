import sys
import theano
import theano.tensor as T
import numpy as np

import lstm
import optimizer
from util import *


flags = None


class Attention:

    @staticmethod
    def bilinear_fn(x, y, W, U, xb, yb):
        """
        @param x: shape (batch_size, n_sent, s0)
        @param y: shape (batch_size, s1)
        @return:  shape (batch_size, s2): batched contexts
        """
        prob = T.batched_dot(T.dot(x, W), y) + T.dot(x, xb) + T.dot(y, yb).dimshuffle(0, 'x')
        context = T.batched_dot(prob, x)               # (batch_size, s0)
        return T.dot(context, U)


    def __init__(self, s0, s1, s2, att_type, pref, pdict):
        """
        s1 should be > s2 to encourage sparsity
        """
        if att_type == 'bilinear':
            self.W = init_matrix_u((s0, s1), pref + '_W', pdict)
            self.U = init_matrix_u((s1, s2), pref + '_U', pdict) 
            self.xb = init_matrix_u((s0,), pref + '_xb', pdict)
            self.yb = init_matrix_u((s1,), pref + '_yb', pdict)
            self.get_params = lambda: [self.W, self.U, self.xb, self.yb]
            self.__call__ = lambda x, y: Attention.bilinear_fn(x, y, self.W, self.U, self.xb, self.yb)
        else:
            assert False


class WordEncoder:

    def __init__(self, pdict, dropout):
        self.rnn = lstm.LSTM(flags['n_layers'], flags['n_embed'], flags['n_hidden'], 
                             dropout, 'wordenc_rnn', pdict)

    def get_params(self):
        return self.rnn.get_params()

    def forward(self, sentence, mask):
        """
        @return: (result, updates)
        multiple sentences can be concatenated and passed in, if seperated by 0 mask 
        """
        hiddens, updates = self.rnn.forward(sentence, mask)
        return hiddens[:, -1], updates


class SentEncoder:

    def __init__(self, pdict, dropout):
        self.word_encoder = WordEncoder(pdict, dropout)
        self.rnn = lstm.LSTM(flags['n_layers'], flags['n_hidden'], flags['n_hidden'], 
                             dropout, 'sentenc_rnn', pdict)
    
    def get_params(self):
        return self.rnn.get_params() + self.word_encoder.get_params()

    def forward(self, concat_sents, concat_masks, doc_sent_pos, doc_mask):
        """
        @param concat_sents: T(*sum_sent_len, *batch_size, n_embed)
        @param concat_masks: T(*sum_sent_len, *batch_size) 
        @param doc_sent_pos: T(*n_sent, *n_doc_batch)
                             doc_sent_pos[j,i] = pos of jth sentence in ith document (in cur batch)
                                                 in all_sent_list/all_mask_list,
                                                 assuming their first two axises are flattened.
        @param doc_mask:     T(*n_sent, *n_doc_batch)
                             mask matrix marking the end of each document
        @return:             (h_sent, updates) where
                              h_sent: T(*n_sent, *n_doc_batch, n_hidden)  
        """

        # == Word-level encoding ==
        sent_embed, upd0 = self.word_encoder.forward(concat_sents, concat_masks) # (ssl, bs, nh)

        # == Reorder ==
        sent_embed_reshaped = T.reshape(sent_embed, [sent_embed.shape[0] * sent_embed.shape[1], 
                                                     sent_embed.shape[2]])
        doc_hidden, _ = theano.scan(sequences=doc_sent_pos,
                                    outputs_info=None,
                                    non_sequences=[sent_embed_reshaped],
                                    fn=lambda batch_pos, sent_embed_rs: sent_embed_rs[batch_pos],
                                    strict=True)
        # doc_hidden.shape ~ [n_sent, n_doc_batch, n_hidden]

        # == Sentence-level == 
        h_encs, upd1 = self.rnn.forward(doc_hidden, doc_mask)
        h_encs = h_encs[:, -1] # Remove all but the highest layer
        return h_encs, concat_updates(upd0, upd1)
        

class WordDecoder:

    def get_params(self):
        return self.rnn.get_params() + [self.W, self.b]

    def __init__(self, embed, pdict, dropout):
        self.rnn = lstm.LSTM(flags['n_layers'], flags['n_embed'], flags['n_hidden'],
                             dropout, 'worddec_rnn', pdict)
        self.embed = embed
        self.W = init_matrix_u((flags['n_hidden'], flags['n_vocab']), 'worddec_W', pdict)
        self.b = init_matrix_u((flags['n_vocab'],), 'worddec_b', pdict)
        self.dropout = dropout
        self.eos = 2 if flags['simplernn'] else 1
    
    def decode(self, h_0, exp_word, exp_mask):
        """
        @param h_0:      T(n_layers, batch_size, n_hidden); last hidden blocks of SentDecoder
        @param exp_word: T(*sent_len, batch_size); expected sentence (batch)
                         exp_word[i][j] = k => expecting token <k>
        @param exp_mask: exp_word.shape[:2]; EOS indicator
        @return:         ((last_hidden, prob), updates)
                         where prob.shape ~ [batch_size]
        """
        n_batch = exp_word.shape[1]
        
        # Forward to RNN
        hiddens, upd_rnn = self.rnn.forward(self.embed[exp_word], exp_mask, h_0=h_0, delta_t=-1)
        hiddens = hiddens[:, -1] # remove all but the last layer

        # Let ruler = [0, n_batch, ..., (n_batch - 1) * n_batch]
        ruler, _ = theano.scan(fn = lambda pre: pre+1,
                               outputs_info = [T.cast(T.alloc(-1), 'int64')],
                               n_steps = n_batch)
        ruler = n_batch * ruler

        # Use top hidden layer to compute probabilities
        def step(h_t, exp_word_t, exp_mask_t, *args):
            """
            @param h_t:      T(n_batch, n_hidden), hidden block of last level
            """
            h_t = self.dropout(h_t)
            act_word = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            prob = T.flatten(act_word)[exp_word_t + ruler] + 1e-6
            log_prob = exp_mask_t * T.log(prob)
            return log_prob
        
        probs, upd_dec = theano.scan(fn=step, 
                                     sequences=[hiddens, exp_word, exp_mask],
                                     outputs_info=None,
                                     non_sequences=[self.W, self.b, ruler] + [self.dropout.switch],
                                     strict=True)

        prob = T.sum(probs, axis=0)
        return (hiddens[-1], prob), concat_updates(upd_rnn, upd_dec)

    def search(self, h_0):
        """
        @param h_0:     np.array of (n_layers, 1, n_hidden)
        @return:        list of best beams [(log-likelihood, last H, token list)] 
        """

        def compile_next():
            c_t = T.ftensor3('c_t')
            h_t = T.ftensor3('h_t')
            x_t = T.fmatrix('x_t')
            xm_t = T.fvector('xm_t')
            c_tp1, h_tp1 = self.rnn.step(x_t, xm_t, c_t, h_t)
            p_word_t = T.log(T.nnet.softmax(T.dot(h_tp1[-1], self.W) + self.b))
            return theano.function([c_t, h_t, x_t, xm_t],
                                   [c_tp1, h_tp1, p_word_t])

        if not hasattr(self, 'rnn_next'):
            self.rnn_next = compile_next()
        
        # == Beam Search ==
        c_0 = 0.0 * h_0
        x_0 = np.zeros((1, flags['n_embed'])).astype('f')
        que = [(0.0, c_0, h_0, x_0, [])]
        final_beams = []
        for i in xrange(flags['n_max_sent']):
            nque = []
            for log_prob, c_t, h_t, x_t, cur_sent in que:
                c_tp1, h_tp1, p_word_t = self.rnn_next(c_t, h_t, x_t, [1.])
                p_word_t = p_word_t.flatten() # (1, n_vocab) -> (n_vocab,)
                tokens_t = np.argpartition(p_word_t, flags['n_beam'])[:flags['n_beam']] # (n_beam,)
                for tok in tokens_t:
                    log_prob_tp1 = log_prob + p_word_t[tok]
                    x_tp1 = self.embed.get_value(borrow=True)[tok].reshape((1, flags['n_embed']))
                    node = (log_prob_tp1, c_tp1, h_tp1, x_tp1, cur_sent + [tok])
                    if tok == self.eos: 
                        final_beams.append(node)
                    else:
                        nque.append(node)
            #
            que = sorted(nque, key=lambda x: -x[0])[:flags['n_beam']]

        final_beams += que
        final_beams = sorted(final_beams, key=lambda x: -x[0])[:flags['n_beam']]
        return [(b[0], b[2], b[4]) for b in final_beams]


class SoftDecoder:

    def get_params(self):
        return self.attention.get_params() + self.rnn.get_params() + \
            self.encoder.get_params() + self.decoder.get_params() + \
            [self.embed]

    def __init__(self):
        # Load parameters if required
        if flags['load_npz'].strip() != "":
            pdict = np.load(flags['load_npz'].strip())
            np.random.set_state(pdict['__np_random_state__'])
        else:
            pdict = None

        # Init dropout
        # NOTE: If we're loading from an existing model,
        # we need to copy random state _after_ the function is compiled.
        self.pdict = pdict
        self.dropout = Dropout(flags['dropout_prob'])

        # Init slave models
        self.encoder = SentEncoder(pdict, self.dropout)
        self.attention = Attention(flags['n_hidden'], flags['n_hidden'], flags['n_context'], 
                                   'bilinear', 'softdec_att', pdict)
        self.rnn = lstm.LSTM(flags['n_layers'], flags['n_hidden'] + flags['n_context'], flags['n_hidden'], 
                             self.dropout, 'softdec_rnn', pdict)
        self.embed = init_matrix_u((flags['n_vocab'], flags['n_embed']), 'embed', pdict)
        self.decoder = WordDecoder(self.embed, pdict, self.dropout)

    def save(self, file_):
        params = self.get_params()
        pdict = {}
        for p in params:
            assert not (p.name in pdict)
            pdict[p.name] = p.get_value()
        pdict['__np_random_state__'] = np.array(np.random.get_state(), dtype='object')
        pdict['__theano_mrg_rstate__'] = np.array(self.dropout.rng.rstate, dtype='object')
        pdict['__theano_mrg_state_updates__'] = np.array(self.dropout.rng.state_updates, dtype='object')
        np.savez(file_, **pdict)

    def train(self, X, Y):
        """
        @param X[0..3]: Input doc info. Check SentEncoder.forward
                        X[0] ~ T(*sum_sent_len, *batch_size) [i64, data]
                        X[1] ~ T(*sum_sent_len, *batch_size) [f32, mask]
                        X[2] ~ T(*n_sent, *n_doc_batch)      [i64, doc_sent_pos]
                        X[3] ~ T(*n_sent, *n_doc_batch)      [f32, doc_mask]
        @param Y:       Expected sentences, where
        @param Y[0]:    T(*n_sent, *sent_len, *n_doc_batch) [i64]
        @param Y[1]:    Mask of (each sentence in) Y[0], shape=Y[0].shape, dtype=f32
                        0 if sentence does not exist (reached EOD)
        @param Y[2]:    Document-level mask of Y[0], shape=[*n_sent, *n_doc_batch]
        @return: (loss, upd for validator, upd for trainer)
        """
        X[0] = self.embed[X[0]]
        
        # == Encoding ==
        h_sentences, upd_enc = self.encoder.forward(*X)
        h_sentences = h_sentences.dimshuffle((1, 0, 2))  # OPTME
        # h_sentences.shape ~ [n_doc_batch, n_sent, n_hid]
        
        # == Sentence-level Decoder == 
        def step(exp_sent, exp_mask, doc_mask, h_dec_tm1, c_dec_tm1, x_dec_tm1, h_encs, *args):
            """
            @exp_sent:  T(*sent_len, *n_doc_batch) [i64]      expected sentence (batch)
            @exp_mask:  shape ~ exp_sent.shape[:-1];          sent-level mask of cur sentence
            @doc_mask:  T(*n_doc_batch,);                     doc-level mask
            @h_dec_tm1: T(n_layers, *n_doc_batch, n_hidden);  previous hidden layers
            @c_dec_tm1: Same shape;                           previous cell states
            @x_dec_tm1: T(*n_doc_batch, n_hidden);            output of last word_decoder
            @h_encs:    T(*n_doc_batch, *n_sent, n_hidden);   h_sentences  
            @*args:     non_sequences that theano should put into GPU 
            """ 
            att_context = self.attention(h_encs, h_dec_tm1[-1]) # (n_doc_batch, n_context)
            inp = T.concatenate([x_dec_tm1, att_context], axis=1)
            c_dec_t, h_dec_t = self.rnn.step(inp, doc_mask, c_dec_tm1, h_dec_tm1) 
            (x_dec_t, prob), upd = self.decoder.decode(h_dec_t, exp_sent, exp_mask) 
            return (h_dec_t, c_dec_t, x_dec_t, prob), upd
        
        batch_size = h_sentences.shape[0]
        scan_params = self.rnn.get_params() + self.attention.get_params() + self.decoder.get_params() \
                + [self.embed, self.dropout.switch] # OPTME: make it elegant
        [h_decs, _, _, probs], upd_dec = \
            theano.scan(fn = step,
                        sequences = Y,
                        outputs_info = [T.alloc(0.0, flags['n_layers'], batch_size, flags['n_hidden']),
                                        T.alloc(0.0, flags['n_layers'], batch_size, flags['n_hidden']),
                                        T.alloc(0.0, batch_size, flags['n_hidden']),
                                        None],
                        non_sequences = [h_sentences] + scan_params,
                        strict=True)
        #

        loss = -T.sum(probs)
        grad = T.grad(-loss, self.get_params()) 
        rng_updates = concat_updates(upd_enc, upd_dec)
        return self.get_params(), loss, grad, rng_updates

    def init_rng(self):
        """
        Restore theano rng state. Must be called _after_ compilation
        """
        if flags['load_npz'].strip() != "":
            self.dropout.rng.rstate = self.pdict['__theano_mrg_rstate__']
            for (su2, su1) in zip(self.dropout.rng.state_updates, self.pdict['__theano_mrg_state_updates__'][0]):
                su2[0].set_value(su1[0].get_value())

