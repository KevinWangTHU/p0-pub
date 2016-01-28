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
        prob = T.batched_dot(T.dot(x, W), y) + T.dot(x, xb) + T.dot(y, yb)
        context = T.batched_dot(prob, x)               # (batch_size, s0)
        return T.dot(context, U)


    def __init__(self, s0, s1, s2, att_type, pref, pdict):
        if att_type == 'bilinear':
            self.W = init_matrix_u((s0, s1), pref + '_W', pdict)
            self.U = init_matrix_u((s1, s2), pref + '_U', pdict) # s1 > s2. Encourages sparsity
            self.xb = init_matrix_u((s0,), pref + '_xb', pdict)
            self.yb = init_matrix_u((s1,), pref + '_yb', pdict)
            self.get_params = lambda: [self.W, self.U, self.xb, self.yb]
            self.fn = lambda x, y: Attention.bilinear_fn(x, y, self.W, self.U, self.xb, self.yb)
        else:
            assert False


class WordEncoder:

    def __init__(self, pdict):
        self.rnn = lstm.LSTM(flags['n_embed'], flags['n_hidden'], 'wordenc', pdict)

    def get_params(self):
        return self.rnn.get_params()

    def forward(self, sentence, mask):
        """
        multiple sentences can be concatenated and passed in, if seperated by 0 mask 
        """
        return self.rnn.forward(sentence, mask)


class SentEncoder:

    def __init__(self, pdict):
        self.word_encoder = WordEncoder(pdict)
        self.rnn = lstm.LSTM(flags['n_hidden'], flags['n_hidden'], 'sentenc', pdict)
    
    def get_params(self):
        return self.rnn.get_params() + self.word_encoder.get_params()

    def forward(self, concat_sents, concat_masks, doc_sent_pos, doc_mask):
        """
        @param concat_sents: T(*sum_sent_len, *batch_size, n_embed)
        @param concat_masks: T(*sum_sent_len, *batch_size) 
        @param doc_sent_pos: T(*n_sent, *doc_batch_size)
                             doc_sent_pos[j,i] = pos of jth sentence in ith document (in cur batch)
                                                 in all_sent_list/all_mask_list,
                                                 assuming their first two axises are flattened.
        @param doc_mask:     T(*n_sent, *doc_batch_size)
                             mask matrix marking the end of each document
        @return:             (h_sent, h_word) where
                              h_sent: T(*n_sent, *doc_batch_size, n_hidden)  
                              h_word: None (for future compatibility)
        """

        # Word-level encoding
        sent_embed = self.word_encoder.forward(concat_sents, concat_masks) # (ssl, bs, nh)

        # Reorder
        sent_embed_reshaped = T.reshape(sent_embed, [sent_embed.shape[0] * sent_embed.shape[1], 
                                                     sent_embed.shape[2]])
        doc_hidden, _ = theano.scan(sequences=doc_sent_pos,
                                    outputs_info=None,
                                    non_sequences=[sent_embed_reshaped],
                                    fn=lambda batch_pos, sent_embed_rs: sent_embed_rs[batch_pos],
                                    strict=True)
        # doc_hidden.shape ~ [n_sent, doc_batch_size, n_hidden]

        # sentence-level RNN
        h_encs = self.rnn.forward(doc_hidden, doc_mask)
        return (h_encs, None)
        

class WordDecoder:

    def get_params(self):
        return self.rnn.get_params() + [self.W, self.b]

    def __init__(self, pdict):
        self.rnn = lstm.LSTM(flags['n_hidden'], flags['n_hidden'], 'worddec_rnn', pdict)
        self.W = init_matrix_u((flags['n_hidden'], flags['n_vocab']), 'worddec_W', pdict)
        self.b = init_matrix_u((flags['n_vocab'],), 'worddec_b', pdict)
    
    def decode(self, h_0, exp_word, exp_mask):
        """
        @param h_0:      T(batch_size, n_hidden)
        @param exp_word: T(*sent_len, batch_size, embed_size); expected sentence (batch)
        @param exp_mask: exp_word.shape[:2]; EOS indicator
        @return:         (last_hidden, prob)
                         where prob.shape ~ [batch_size]
        """
        def step(exp_word, exp_mask, c_tm1, h_tm1, x_tm1, *args):
            """
            @param exp_word:     T(batch_size, embed_size)
            @param exp_mask:     T(batch_size,)
            @param c_tm1, h_tm1: T(batch_size, hidden_size)
            """
            c_t, h_t = self.rnn.step(x_tm1, exp_mask, c_tm1, h_tm1)
            act_word = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            prob = exp_mask * T.sum(exp_word * T.log(act_word + 1e-6), axis=1)
            return c_t, h_t, exp_word, prob
        
        batch_size = h_0.shape[0]
        [_, h_decs, _, probs], _ = \
            theano.scan(fn=step,
                        sequences=[exp_word, exp_mask],
                        outputs_info=[T.alloc(0.0, batch_size, flags['n_hidden']),
                                      h_0,
                                      T.alloc(0.0, batch_size, flags['n_embed']),
                                      None],
                        non_sequences=self.get_params(),
                        strict=True)

        prob = T.sum(probs, axis=0)
        return h_decs[-1], prob


class SoftDecoder:

    def get_params(self):
        return self.att.get_params() + self.rnn.get_params() + \
            self.encoder.get_params() + self.decoder.get_params() + \
            [self.embed]

    def __init__(self):
        if flags['load_npz'].strip() != "":
            pdict = np.load(flags['load_npz'].strip())
            np.random.set_state(pdict['__np_random_state__'])
        else:
            pdict = None

        self.encoder = SentEncoder(pdict)
        self.decoder = WordDecoder(pdict)
        self.att = Attention(flags['n_hidden'], flags['n_hidden'], flags['n_context'], 
                             'bilinear', 'mn_sft_att', pdict)
        self.rnn = lstm.LSTM(flags['n_hidden'] + flags['n_context'], flags['n_hidden'], 
                             'rnn_doc_dec', pdict)
        self.embed = init_matrix_u((flags['n_vocab'], flags['n_hidden']), 'word_embedding', pdict)

    def save(self, file_):
        params = self.get_params()
        pdict = {}
        for p in params:
            assert not (p.name in pdict)
            pdict[p.name] = p.get_value()
        pdict['__np_random_state__'] = np.random.get_state()
        np.savez(file_, **pdict)

    def train(self, X, Y):
        """
        @param X[0..3]: Input doc info. Check SentEncoder.forward
                        X[0] ~ T(*sum_sent_len, *batch_size) [i64, data]
                        X[1] ~ T(*sum_sent_len, *batch_size) [f32, mask]
                        X[2] ~ T(*n_sent, *doc_batch_size)   [i64, doc_sent_pos]
                        X[3] ~ T(*n_sent, *doc_batch_size)   [f32, doc_mask]
        @param Y:       Expected sentences, where
        @param Y[0]:    T(*n_sent, *sent_len, *doc_batch_size) [i64]
        @param Y[1]:    Mask of (each sentence in) Y[0], shape=Y[0].shape, dtype=f32
                        0 if sentence does not exist (reached EOD)
        @param Y[2]:    Document-level mask of Y[0], shape=[*n_sent, *doc_batch_size]
        @return: (update dict, loss)
        """
        X[0] = self.embed[X[0]]
        Y[0] = self.embed[Y[0]]

        h_sentences, _ = self.encoder.forward(*X)
        h_sentences = h_sentences.dimshuffle((1, 0, 2))  # OPTME
        # ~ [d_b_s, n_s, n_h]
        
        # Decoder LSTM with attention
        def step(exp_sent, exp_mask, doc_mask, h_dec_tm1, c_dec_tm1, x_dec_tm1, h_encs, *args):
            """
            @exp_sent:  T(*sent_len, *doc_batch_size, n_embed);  expected sentence (batch)
            @exp_mask:  shape ~ exp_sent.shape[:-1];             sent-level mask of cur sentence
            @doc_mask:  T(*doc_batch_size,);                     doc-level mask
            @h_dec_tm1: T(*doc_batch_size, n_hidden);            previous hidden layer
            @c_dec_tm1: Same shape;                              previous cell state
            @x_dec_tm1: Same shape;                              last hidden of last word_decoder
            @h_encs:    T(*doc_batch_size, *n_sent, n_hidden);   h_sentences  
            @*args:     non_sequences that theano should put into GPU 
            """ 
            # Soft attention
            att_context = self.att.fn(h_encs, h_dec_tm1) # (doc_batch_size, n_context)
            inp = T.concatenate([x_dec_tm1, att_context], axis=1)
            c_dec_t, h_dec_t = self.rnn.step(inp, doc_mask, c_dec_tm1, h_dec_tm1) 
            x_dec_t, prob = self.decoder.decode(h_dec_t, exp_sent, exp_mask) 
            return h_dec_t, c_dec_t, x_dec_t, prob
        
        batch_size = h_sentences.shape[0]
        scan_params = self.rnn.get_params() + self.att.get_params() + self.decoder.get_params()
        [h_decs, _, _, probs], _ = \
            theano.scan(fn = step,
                        sequences = Y,
                        outputs_info = [T.alloc(0.0, batch_size, flags['n_hidden']),
                                        T.alloc(0.0, batch_size, flags['n_hidden']),
                                        T.alloc(0.0, batch_size, flags['n_hidden']),
                                        None],
                        non_sequences = [h_sentences] + scan_params,
                        strict=True)
        loss = -T.sum(probs)
        grad = T.grad(-loss, self.get_params()) 
        grad_updates = optimizer.optimize(flags['optimizer'], self.get_params(), grad, {}, flags)
        return loss, grad_updates

