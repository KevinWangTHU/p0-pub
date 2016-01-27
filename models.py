"""
# TODO
"""

import theano
import theano.tensor as T
import numpy as np

import lstm
import optimizer
from util import *

import gflags

flags = gflags.FLAGS

class Attention:

    """
    @fn: (x: Tensor) (y: Tensor) -> Tensor
        @param x: shape (batch_size, n_sent, s0)
        @param y: shape (batch_size, s1)
        @return:  shape (batch_size, n_sent,); attention strength
    """

    def __init__(self, s0, s1, att_type, pref):
        if att_type == 'bilinear':
            W = init_matrix_u((s0, s1), pref + '_att_W')
            xb = init_matrix_u((s0,), pref + '_att_xb')
            yb = init_matrix_u((s1,), pref + '_att_yb')
            self.get_params = lambda: [W, xb, yb]
            self.fn = lambda x, y: \
                T.batched_dot(T.dot(x, W), y) + T.dot(x, xb) + T.dot(y, yb) # will be broadcasted.
        else:
            assert False


class WordEncoder:

    rnn = None

    def __init__(self):
        rnn = lstm.LSTM(flags['n_emb'], flags['n_hidden'], 'wordenc')

    def get_params(self):
        return self.rnn.get_params()

    def encode(self, sentence, mask):
        """
        @param sentence: T(*sent_len, batch_size, embed_size)
        @param mask:     T(*sent_len, batch_size)  
        @return:         T(*sent_len, batch_size, n_hidden)
        """
        return self.rnn.forward(sentence, mask)


class SentEncoder:

    word_encoder = None
    rnn = None
    n_hidden = None

    def __init__(self):
        word_encoder = WordEncoder(flags)
        n_hidden = flags['n_hidden']
        rnn = lstm.LSTM(flags['n_hidden'], flags['n_hidden'], 'sentenc')
    
    def get_params(self):
        return self.rnn.get_params() + self.word_encoder.get_params()

    def batch_word_encode(self, all_sent_list, all_mask_list):
        """
        @return: T(n_batches, batch_size, n_hidden)
        """
        def step(sent_batch, mask_batch, *args):
            return self.word_encoder.encode(sent_batch, mask_batch)[-1]
        # 
        code, _ = theano.scan(fn=step,
                              sequences=[all_sent_list, all_mask_list],
                              outputs_info=T.alloc(0.0, batch_size, self.n_hidden),
                              non_sequences=self.word_encoder.get_params())
        return code

    def forward(self, all_sent_list, all_mask_list, doc_sent_pos, doc_mask):
        """
        @param all_sent_list: T(*n_batches, *sent_len, *batch_size, n_embed)
        @param all_mask_list: T(*n_batches, *sent_len, *batch_size) (Int)
        @param doc_sent_pos:  T(*n_sent, *doc_batch_size)
                              doc_sent_pos[i,j] = pos of jth sentence in ith document (in cur batch)
                                                  in all_sent_list/all_mask_list,
                                                  assuming their first two axises are flattened.
        @param doc_mask:      T(*n_sent, *doc_batch_size)
                              mask matrix marking the end of each document
        @return:              (h_sent, h_word) where
                               h_sent: T(*n_sent, *doc_batch_size, n_hidden)  
                               h_word: None (for future compatibility)
        """

        # == Word-level encoding ==
        sent_embed = self.batch_word_encode(all_sent_list, doc_sent_pos)
        n_hidden = sent_embed.shape[2]

        # == Reshape ==
        sent_embed_reshaped = T.reshape(sent_embed, [sent_embed.shape[0] * sent_embed.shape[1]] + sent_embed.shape[2:])
        doc_hidden = theano.scan(sequences=doc_sensent_embed_pos,
                                 outputs_info=None,
                                 non_sequences=sent_embed_reshaped,
                                 fn=lambda batch_pos, sent_embed_rs: sent_embed_rs[batch_pos])
        # doc_hidden.shape ~ [n_sent, doc_batch_size, n_hidden]

        # == sentence-level RNN ==
        h_encs = self.rnn.forward(doc_hidden, doc_mask)
        return (h_encs, None)
        

class WordDecoder:

    rnn = None
    W = None
    b = None

    def get_params(self):
        return self.rnn.get_params() + [self.W, self.b]

    def __init__(self):
        self.rnn = LSTM.lstm(flags['n_hidden'], flags['n_hidden'], 'worddec_rnn')
        W = init_matrix_u((flags['n_hidden'], flags['n_words']), 'worddec_W')
        b = init_matrix_u((flags['n_words'],), 'worddec_b')
    
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
            c_t, h_t = rnn.step(x_tm1, exp_mask, c_tm1, h_tm1)
            act_word = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            prob = exp_mask * T.sum(exp_word * T.log(act_word + 1e-6), axis=1)
            return c_t, h_t, exp_word, prob

        [_, h_decs, _, probs], _ = \
            theano.scan(fn=step,
                        sequences=[exp_word, exp_mask],
                        outputs_info=[T.alloc(0.0, batch_size, flags['n_hidden']),
                                      T.alloc(0.0, batch_size, flags['n_hidden']),
                                      T.alloc(0.0, batch_size, flags['n_embed']),
                                      None],
                        non_sequences=self.get_params())

        prob = T.sum(probs, axis=0)
        return h_decs[-1], prob


class SoftDecoder:

    encoder = None # Sentence-level Encoder
    decoder = None # Word-level Decoder
    rnn = None     # Sentence-level RNN for decoding
    att = None     # Attention for self.rnn

    def get_params(self):
        return self.att.get_params() + self.rnn.get_params() + \
            self.encoder.get_params() + self.decoder.get_params()

    def __init__(self):
        self.encoder = SentEncoder(flags)
        self.decoder = WordDecoder(flags)
        self.att = Attention(flags['n_hidden'], flags['n_hidden'], 'bilinear', 'mn_sft_att')
        self.rnn = lstm.LSTM(flags['n_hidden'], flags['n_hidden'], 'rnn_doc_dec')

    def train(self, X, Y):
        """
        @param X[0..3]: Input doc info. Check SentEncoder.forward.
        @param Y:       Expected sentences, where
        @param Y[0]:    T(*n_sent, *sent_len, *doc_batch_size, n_embed)
        @param Y[1]:    Mask of (each sentence in) Y[0], shape=Y[0].shape[:-1]
                        0 if sentence does not exist (reached EOD)
        @param Y[2]:    Document-level mask of Y[0], shape=[*n_sent, *doc_batch_size]
        @return: (update dict, loss)
        """
        h_sentences, _ = self.encoder.forward(X[0], X[1], X[2], X[3])
        h_sentences = h_sentences.dimshuffle((1, 0, 2))  # OPTME
        
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
            att_prob = T.nnet.softmax(self.att.fn(h_encs, h_dec_tm1)) # (batch_size, n_sent) 
            att_context = T.batched_dot(att_prob, h_encs)               # (batch_size, hidden_size)
            inp = T.concatenate([x_dec_tm1, att_context], axis=1)
            c_dec_t, h_dec_t = self.rnn.step(inp, doc_mask, c_dec_tm1, h_dec_tm1) 
            x_dec_t, prob = word_decoder.decode(h_dec_t, exp_sent, exp_mask) 
            return h_dec_t, c_dec_t, x_dec_t, prob
        
        scan_params = self.rnn.get_params() + self.att.get_params() + self.decoder.get_params()
        [h_decs, c_decs, x_decs, probs], _ = \
            theano.scan(fn = step,
                        sequences = [Y[0], Y[1], Y[2]],
                        outputs_info = [T.alloc(0.0, batch_size, hidden_size),
                                        T.alloc(0.0, batch_size, hidden_size),
                                        T.alloc(0.0, batch_size, hidden_size),
                                        None],
                        non_sequences = [h_sentences] + scan_params)

        loss = -T.sum(probs)
        grad = T.grad(-loss, self.get_params())
        grad_updates = optimizer.optimize('RMSProp2', self.get_params(), grad, {}, flags)


