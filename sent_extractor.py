import theano
import theano.tensor as T
import numpy as np

import lstm
from util import *
import models

flags = None

class SentExtractor:
    
    def get_params(self):
        return self.attention.get_params() + self.rnn.get_params() + self.encoder.get_params() + [self.embed]
    
    def __init__(self):
        global flags
        flags = models.flags
        # Load parameters if required
        if flags['load_npz'].strip() != "":
            pdict = np.load(flags['load_npz'].strip())
            np.random.set_state(pdict['__np_random_state__'])
        else:
            pdict = None
        if (pdict and not 'embed' in pdict) and flags['wordvec']: # Use pre-trained wordvec.
            with open(flags['wordvec']) as fin:
                pdict['embed'] = cPickle.load(fin)
            
        self.pdict = pdict
        self.dropout = Dropout(flags['dropout_prob'])
        
        # Slave models
        n_att_input = flags['n_hidden'] + 1 if flags['attend_pos'] else flags['n_hidden']
        self.attention = models.Attention(n_att_input, flags['n_hidden'], flags['n_context'], 
                                   'bilinear', 'softdec_att', pdict, no_context=True)
        self.encoder = models.SentEncoder(pdict, self.dropout)
        self.word_encoder = self.encoder.word_encoder
        self.rnn = lstm.LSTM(flags['n_layers'], flags['n_hidden'], flags['n_hidden'], 
                             self.dropout, 'softdec_rnn', pdict)
        
        # Word embedding
        self.embed = init_matrix_u((flags['n_vocab'], flags['n_embed']), 'embed', pdict)
        
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
        
    def att_step_x(self, exp_sent, exp_mask, doc_mask, scores_t, dropout_mask, h_dec_tm1, c_dec_tm1, h_encs, *args):
        """
        Step function for objective A: max E_{s[t]~p_att(s|highlights[t-1])}[ROUGE(s[t])]
        @exp_sent:     ftensor3(*sent_len, *n_doc_batch, n_embed)
                                                             expected highlights (batch)
        @scores_t:     fmatrix((*n_doc_batch, *n_sent));     scores[t]
        @exp_mask, doc_mask, dropout_mask, h_dec_tm1, c_dec_tm1, h_encs: Check SoftDecoder
        """ 
        att_prob = self.attention.get_prob(h_encs, h_dec_tm1[-1]) # fmat(n_doc_batch, n_sent)
        exp_reward = T.batched_dot(att_prob, scores_t) # fvec(n_doc_batch)
        x_dec_t, upd_0 = self.word_encoder.forward(exp_sent, exp_mask)
        x_dec_t = x_dec_t[-1]
        c_dec_t, h_dec_t = self.rnn.step(x_dec_t, doc_mask, dropout_mask, c_dec_tm1, h_dec_tm1)
        return (h_dec_t, c_dec_t, exp_reward, batch_entropy(att_prob)), upd_0
    
    def att_step_y(self, exp_sent, exp_mask, doc_mask, scores_t, dropout_mask, h_dec_tm1, c_dec_tm1, h_encs, *args):
        """
        Step function for objective B: max E_{s[t]~p_att(s|s[t-1])}[ROUGE(s[t])]
        """
        att_prob = self.attention.get_prob(h_encs, h_dec_tm1[-1]) # fmat(n_doc_batch, n_sent)
        exp_reward = T.batched_dot(att_prob, scores_t) # fvec(n_doc_batch)
        x_dec_t = T.batched_dot(att_prob, h_encs)
        if flags['attend_pos']: # remove position info. Check SentEncoder.forward
            x_dec_t = x_dec_t[:, :flags['n_hidden']]
        c_dec_t, h_dec_t = self.rnn.step(x_dec_t, doc_mask, dropout_mask, c_dec_tm1, h_dec_tm1)
        return h_dec_t, c_dec_t, exp_reward, batch_entropy(att_prob)
     
    def train(self, X, Y, score, alpha):
        """
        @param X[0..3]: Input doc info. Check SentEncoder.forward
        @param Y[0..2]: Expected sentences
        @param score:   ftensor3((*n_hl_sent, *n_doc_batch, *n_sent)),
                        score[k, j, i]: similarity between sent[j, i] and highlight[j, k]
        @param alpha:   fscalar, mix weight of two objective functions.
        """
        X[0] = self.embed[X[0]]
        Y[0] = self.embed[Y[0]]
        
        # == Encoding ==
        h_sentences, lasthid_enc, upd_enc = self.encoder.forward(*X)
        h_sentences = h_sentences.dimshuffle((1, 0, 2))
        # h_sentences.shape ~ [*n_doc_batch, n_sent, n_hid]
        # lasthid_enc.shape ~ [n_layer, n_doc_batch, n_hid]
        
        batch_size = h_sentences.shape[0]
        def scan(step_fn):
            scan_params = self.rnn.get_params() + self.attention.get_params() + self.word_encoder.get_params() \
                    + [self.embed, self.dropout.switch] 
            dropout_masks = self.dropout.prep_mask(
                    (Y[0].shape[0], flags['n_layers'], batch_size, flags['n_hidden']))
            return theano.scan(
                fn = step_fn,
                sequences = Y + [score, dropout_masks],
                outputs_info = [
                    lasthid_enc, # H
                    T.alloc(0.0, flags['n_layers'], batch_size, flags['n_hidden']), # C
                    None, None],
                non_sequences = [h_sentences] + scan_params,
                strict=True)
        
        [_, _, reward_x, ent_x], upd_x = scan(lambda *args: self.att_step_x(*args))
        [_, _, reward_y, ent_y], upd_y = scan(lambda *args: self.att_step_y(*args))
        obj = alpha * T.sum(reward_x) + (1 - alpha) * T.sum(reward_y)
        obj_ent = (alpha * T.sum(ent_x) + (1 - alpha) * T.sum(ent_y)) * flags['w_entropy']
        obj = theano.printing.Print('obj')(T.mean(obj))
        obj_ent = theano.printing.Print('obj_ent')(T.mean(obj_ent))
        obj += obj_ent
        loss = -obj / T.cast(batch_size, 'float32')
        grad = T.grad(-loss, self.get_params()) 
        rng_updates = upd_x + upd_y
        
        return self.get_params(), loss, grad, rng_updates

    def init_rng(self):
        """
        Restore theano rng state. Must be called _after_ compilation
        """
        if flags['load_npz'].strip() != "":
            self.dropout.rng.rstate = self.pdict['__theano_mrg_rstate__']
            for (su2, su1) in zip(self.dropout.rng.state_updates, self.pdict['__theano_mrg_state_updates__'][0]):
                su2[0].set_value(su1[0].get_value())

