import sys
import theano
import theano.tensor as T
import numpy as np
import cPickle

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
        prob = T.nnet.softmax(prob)
        context = T.batched_dot(prob, x)               # (batch_size, s0)
        return T.dot(context, U), prob


    def __init__(self, s0, s1, s2, att_type, pref, pdict):
        """
        s1 should be > s2 to encourage sparsity
        """
        if att_type == 'bilinear':
            self.W = init_matrix_u((s0, s1), pref + '_W', pdict)
            self.U = init_matrix_u((s0, s2), pref + '_U', pdict) 
            self.xb = init_matrix_u((s0,), pref + '_xb', pdict)
            self.yb = init_matrix_u((s1,), pref + '_yb', pdict)
            self.get_params = lambda: [self.W, self.U, self.xb, self.yb]
            self.__call__ = lambda x, y: Attention.bilinear_fn(x, y, self.W, self.U, self.xb, self.yb)[0]
            self.get_prob = lambda x, y: Attention.bilinear_fn(x, y, self.W, self.U, self.xb, self.yb)[1]
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
        hiddens, updates, _ = self.rnn.forward(sentence, mask)
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
        @return:             (h_sent, last_hid, updates) where
        @h_sent:             T(*n_sent, *n_doc_batch, n_hidden[+1]), encoding of all sentences in each doc
        @last_hid:           T(n_layers, n_doc_batch, n_hidden), last hidden block of sentence-level RNN
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
        h_encs, upd1, last_hid = self.rnn.forward(doc_hidden, doc_mask)

        if flags['attend_pos']:
            # Append position info in each hidden vec
            ruler = gen_ruler(doc_hidden.shape[0]) + 1
            doc_ruler = (doc_mask.T * ruler) / T.cast(T.shape_padright(doc_mask.sum(axis=0)), 'float32')
            doc_ruler = T.cast(doc_ruler.T, 'float32') # (n_sent, n_doc_batch)
            doc_hidden = T.concatenate([doc_hidden, T.shape_padright(doc_ruler)], axis=2)

        return doc_hidden, last_hid, concat_updates(upd0, upd1)
        # return h_encs[:, -1], T.zeros_like(last_hid), concat_updates(upd0, upd1)
        

class WordDecoder:

    def get_params(self):
        return self.rnn.get_params() + [self.W0, self.b0, self.b]

    def __init__(self, embed, pdict, dropout):
        self.rnn = lstm.LSTM(flags['n_layers'], flags['n_embed'], flags['n_hidden'],
                             dropout, 'worddec_rnn', pdict)
        self.embed = embed
        # p(word|hid) = softmax((hid . W0 + b0) . embed.T + b)
        self.W0 = init_matrix_u((flags['n_hidden'], flags['n_embed']), 'worddec_W0', pdict)
        self.b0 = init_matrix_u((flags['n_embed'],), 'worddec_b0', pdict)
        self.b = init_matrix_u((flags['n_vocab'],), 'worddec_b', pdict)
        self.dropout = dropout
        self.eos = 2 if flags['simplernn'] else 1
    
    def decode(self, h_0, exp_word, exp_mask, output_words): # TODO: output_words
        """
        @param h_0:      T(n_layers, batch_size, n_hidden); last hidden blocks of SentDecoder
        @param exp_word: T(*sent_len, batch_size); expected sentence (batch)
                         exp_word[i][j] = k => expecting token <output_words[k]>
        @param exp_mask: exp_word.shape[:2]; EOS indicator
        @param output_words:
                         shared(lvector(n_out_vocab*)), allowed words for output
                         must be [0, 1, ..., n_vocab - 1] when testing
                         <None>: the model will only predict first flag['n_out_vocab'] words
        @return:         ((last_hidden, prob), updates)
                         where prob.shape ~ [batch_size]
        """

        n_batch = exp_word.shape[1]
        
        # Forward to RNN
        hiddens, upd_rnn, last_hid = self.rnn.forward(self.embed[exp_word], exp_mask, h_0=h_0, delta_t=-1)
        hiddens = hiddens[:, -1] # remove all but the last layer

        # Let ruler = [0, n_vocab*, ..., (n_batch - 1) * n_vocab*]
        if output_words:
            ruler = output_words.shape[0] * gen_ruler(n_batch)
        else:
            ruler = flags['n_out_vocab'] * gen_ruler(n_batch)

        # Use top hidden layer to compute probabilities
        def step(h_t, exp_word_t, exp_mask_t, dropout_mask, embed_l, b_l, *args):
            """
            @param h_t:      T(n_batch, n_hidden), hidden block of last level
            """
            h_t = self.dropout(h_t, dropout_mask)
            act_word = T.nnet.softmax(T.dot(T.dot(h_t, self.W0) + self.b0, embed_l.T) + b_l)
            prob = T.flatten(act_word)[exp_word_t + ruler] + 1e-6
            log_prob = exp_mask_t * T.log(prob)
            return log_prob
        
        if output_words: 
            embed_l, b_l = self.embed[output_words], self.b[output_words]
        else:
            embed_l, b_l = self.embed, self.b # l for local

        dropout_masks = self.dropout.prep_mask(
                (hiddens.shape[0], hiddens.shape[1], hiddens.shape[2]))
        probs, upd_dec = theano.scan(
                fn=step, 
                sequences=[hiddens, exp_word, exp_mask, dropout_masks],
                outputs_info=None,
                non_sequences=[embed_l, b_l, self.W0, self.b0, self.b, ruler] + [self.dropout.switch],
                strict=True)

        prob = T.sum(probs, axis=0)
        return (last_hid, prob), concat_updates(upd_rnn, upd_dec)

    def search_b(self, h_0, n_max_sent, prune_res=True):
        """
        @param h_0:     np.array of (n_layers, batch_size, n_hidden)
        @return:        list of best beams [(NLL, last_hid, token list)] for each input in batch
                        last_hid.shape ~ (n_layers, n_hidden)
        """

        def compile_next():
            """
            compiles LSTM step function
            @return:    c_tp1, h_tp1, p_word_t ~ (batch_size, n_vocab)
            """
            c_t = T.ftensor3('c_t')
            h_t = T.ftensor3('h_t')
            x_t = T.fmatrix('x_t')
            xm_t = T.fvector('xm_t')
            dropout_mask = (1.0 - self.dropout.prob) * \
                    T.ones((h_t.shape[0], h_t.shape[1], max(flags['n_hidden'], flags['n_embed'])))
            c_tp1, h_tp1 = self.rnn.step(x_t, xm_t, dropout_mask, c_t, h_t)
            h_tp1c = self.dropout(h_tp1, dropout_mask)
            p_word_t = T.nnet.softmax(T.dot(T.dot(h_tp1c[-1], self.W0) + self.b0, self.embed.T) + self.b)
            # p_word_t = T.log(T.nnet.softmax(T.dot(h_tp1c[-1], T.dot(self.W0, self.W1)) + self.b))
            return theano.function([c_t, h_t, x_t, xm_t],
                                   [c_tp1, h_tp1, p_word_t])

        if not hasattr(self, 'rnn_next'):
            self.rnn_next = compile_next()

        # == Batched Beam Search ==
        def sel(arr, i):
            return arr[:, i].reshape((arr.shape[0], 1) + arr.shape[2:])

        n_input_batch = h_0.shape[1]
        c_0 = 0.0 * h_0
        x_0 = np.zeros((n_input_batch, flags['n_embed'])).astype('f')
        que = [[(0., sel(c_0, i), sel(h_0, i), x_0[i], [], i)]\
               for i in xrange(n_input_batch)]
        results = [[] for i in xrange(n_input_batch)]

        for t in xrange(n_max_sent):
            nll_list, c_t_list, h_t_list, x_t_list, tokens_list, id_list = zip(*concat(que)) 
            c_t = np.concatenate(c_t_list, axis=1)
            h_t = np.concatenate(h_t_list, axis=1)
            x_t = np.vstack(x_t_list)
            xm_t = np.ones((x_t.shape[0], )).astype('f')
            # Forward batch to RNN
            c_tp1, h_tp1, p_word_t = self.rnn_next(c_t, h_t, x_t, xm_t)
            argp_t = np.argpartition(p_word_t, -(flags['n_beam'] + 1))[:, -(flags['n_beam'] + 1):] # +1 for EOS
            beam_candidates = [[] for _ in xrange(n_input_batch)]
            # For each beam in que, construct new beams
            for bpos in xrange(len(c_t_list)):
                inp_id = id_list[bpos]
                for tok in argp_t[bpos]:
                    nll_tp1 = nll_list[bpos] - p_word_t[bpos, tok]
                    if tok == self.eos:
                        results[inp_id].append((nll_tp1, h_tp1[:, bpos], tokens_list[bpos] + [tok]))
                    else:
                        beam_candidates[inp_id].append((nll_tp1, bpos, tok))
            # Keep top `n_beam` beams for each input
            beam_candidates = [sorted(q)[:flags['n_beam']] for q in beam_candidates]
            # Append c, h, x
            for i, beams_i in enumerate(beam_candidates):
                for j, (nll, bpos, tok) in enumerate(beams_i):
                    x_tp1 = self.embed.get_value(borrow=True)[tok].reshape((flags['n_embed'],))
                    n_tokens = tokens_list[bpos] + [tok]
                    beam_candidates[i][j] = (nll, sel(c_tp1, bpos), sel(h_tp1, bpos), x_tp1, n_tokens, i)
            que = beam_candidates
        
        # Clean que
        keep = flags['n_beam'] if prune_res else None
        nque = [[(q[0], q[2], q[4]) for q in que_i] for que_i in que]
        results = [sorted(res_i + nq_i)[:keep] for res_i, nq_i in zip(results, nque)]
        return results


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
        n_att_input = flags['n_hidden'] + 1 if flags['attend_pos'] else flags['n_hidden']
        self.attention = Attention(n_att_input, flags['n_hidden'], flags['n_context'], 
                                   'bilinear', 'softdec_att', pdict)
        self.encoder = SentEncoder(pdict, self.dropout)
        self.rnn = lstm.LSTM(flags['n_layers'], flags['n_hidden'] + flags['n_context'], flags['n_hidden'], 
                             self.dropout, 'softdec_rnn', pdict)

        if not 'embed' in pdict and flags['wordvec']: # Use pre-trained wordvec.
            with open(flags['wordvec']) as fin:
                pdict['embed'] = cPickle.load(fin)
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

    def att_step_rnn(self, doc_mask, dropout_mask, h_dec_tm1, c_dec_tm1, x_dec_tm1, h_encs):
        # Code folding
        att_context = self.attention(h_encs, h_dec_tm1[-1]) # (n_doc_batch, n_context)
        inp = T.concatenate([x_dec_tm1, att_context], axis=1)
        c_dec_t, h_dec_t = self.rnn.step(inp, doc_mask, dropout_mask, c_dec_tm1, h_dec_tm1) 
        return c_dec_t, h_dec_t

    def att_step(self, exp_sent, exp_mask, doc_mask, dropout_mask, h_dec_tm1, c_dec_tm1, x_dec_tm1, h_encs, output_words, *args):
        """
        Step function in sentence-level decoding.
        @exp_sent:     T(*sent_len, *n_doc_batch) [i64]      expected sentence (batch)
        @exp_mask:     shape ~ exp_sent.shape[:-1];          sent-level mask of cur sentence
        @doc_mask:     T(*n_doc_batch,);                     doc-level mask
        @dropout_mask: T(n_layers, *n_doc_batch, n_hidden + n_context)         
        @h_dec_tm1:    T(n_layers, *n_doc_batch, n_hidden);  previous hidden layers
        @c_dec_tm1:    Same shape;                           previous cell states
        @x_dec_tm1:    T(*n_doc_batch, n_hidden);            output of last word_decoder
        @h_encs:       T(*n_doc_batch, *n_sent, n_hidden[+1]);   h_sentences  
        @output_words: shared(lvector(n_out_vocab*));        allowed output vocab
        """ 
        output_words = output_words if self.use_output_words else None # Otherwise it will be what should be *args[0]
        c_dec_t, h_dec_t = self.att_step_rnn(doc_mask, dropout_mask, h_dec_tm1, c_dec_tm1, x_dec_tm1, h_encs)
        (last_hidden, prob), upd = self.decoder.decode(h_dec_t, exp_sent, exp_mask, output_words) 
        x_dec_t = last_hidden[-1]
        return (h_dec_t, c_dec_t, x_dec_t, prob), upd

    def train(self, X, Y, output_words):
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
        @output_words:  shared(lvector(n_out_vocab*));        allowed output vocab
        @return: (loss, upd for validator, upd for trainer)
        """
        self.use_output_words = True if output_words else False
        X[0] = self.embed[X[0]]
        
        # == Encoding ==
        h_sentences, lasthid_enc, upd_enc = self.encoder.forward(*X)
        h_sentences = h_sentences.dimshuffle((1, 0, 2))  # OPTME
        # h_sentences.shape ~ [*n_doc_batch, n_sent, n_hid]
        # lasthid_enc.shape ~ [n_layer, n_doc_batch, n_hid]
        
        batch_size = h_sentences.shape[0]
        scan_params = self.rnn.get_params() + self.attention.get_params() + self.decoder.get_params() \
                + [self.embed, self.dropout.switch] 
        if output_words:
            scan_params = [output_words] + scan_params
        dropout_masks = self.dropout.prep_mask(
                (Y[0].shape[0], flags['n_layers'], batch_size, flags['n_hidden'] + flags['n_context']))
        [h_decs, _, _, probs], upd_dec = theano.scan(
            fn = lambda *args: self.att_step(*args),
            sequences = Y + [dropout_masks],
            outputs_info = [
                lasthid_enc, # H
                T.alloc(0.0, flags['n_layers'], batch_size, flags['n_hidden']), # C
                T.alloc(0.0, batch_size, flags['n_hidden']), # X
                None],
            non_sequences = [h_sentences] + scan_params,
            strict=True)
        #

        loss = -T.sum(probs) / T.cast(batch_size, 'float32')
        grad = T.grad(-loss, self.get_params()) 
        rng_updates = concat_updates(upd_enc, upd_dec)
        return self.get_params(), loss, grad, rng_updates

    def test(self, X, max_doc_len, max_sent_len):
        """
        @param X[]: Input representing *one* document. numpy.ndarray of the same shape as .train
        @return:    Generated highlights as 
                    [[(negative-log-likelihood, [token-list]) for each beam] for each doc]
        """
        def compile_aux():
            self.aux_compiled = True

            # = forward =
            # may consider make h_sentences a shared variable btw the two functions
            args = [T.lmatrix('X_data'), T.fmatrix('X_mask'), T.lmatrix('X_pos'), T.fmatrix('X_mask_d')]
            X = [a for a in args]
            X[0] = self.embed[X[0]]
            h_sentences, last_hid, upd_enc = self.encoder.forward(*X)
            h_sentences = h_sentences.dimshuffle((1, 0, 2)) 
            self.f_forward = theano.function(args, (h_sentences, last_hid), updates=upd_enc)

            # = att_step_rnn =
            step_args = [T.ftensor3('h_dec_tm1'), T.ftensor3('c_dec_tm1'), T.fmatrix('x_dec_tm1'),
                         T.ftensor3('h_encs')]
            n_batch = step_args[0].shape[1]
            doc_mask = T.ones((n_batch,), dtype='float32')
            dropout_mask = (1.0 - self.dropout.prob) * \
                T.ones((flags['n_layers'], n_batch, flags['n_hidden'] + flags['n_context']))
            c_dec_t, h_dec_t = self.att_step_rnn(doc_mask, dropout_mask, *step_args)
            att_distro = self.attention.get_prob(step_args[3], step_args[0][-1]) # (n_doc_batch, n_context)
            self.f_att_step_rnn = theano.function(step_args, [c_dec_t, h_dec_t, att_distro])

        # == test(self, X): ==
        if not hasattr(self, 'aux_compiled'):
            compile_aux()

        h_encs, lasthid_enc = self.f_forward(*X)

        # == Beam search ==
        n_batch = lasthid_enc.shape[1]
        assert n_batch == 1
        que = [(0.0, lasthid_enc, 0.0 * lasthid_enc, np.zeros((n_batch, flags['n_hidden'])).astype('f'), [], [])]
        for t in xrange(max_doc_len):
            nque = []
            for nll_tm1, h_tm1, c_tm1, x_t, tokens_tm1, att_tm1 in que:
                c_t, h_t, att_t = self.f_att_step_rnn(h_tm1, c_tm1, x_t, h_encs)
                candidates = self.decoder.search_b(h_t, max_sent_len, prune_res=True)
                for nll_sent, lasthid_sent, tokens_sent in candidates[0]:
                    nque.append((nll_tm1 + nll_sent, h_t, c_t, lasthid_sent[-1].reshape((1, flags['n_hidden'])), tokens_tm1 + [tokens_sent], att_tm1 + [att_t]))
            nque = sorted(nque)[:flags['n_beam']]
            que = nque

        ret = [[(elem[0], elem[4], elem[5]) for elem in que]]
        return ret

    def init_rng(self):
        """
        Restore theano rng state. Must be called _after_ compilation
        """
        if flags['load_npz'].strip() != "":
            self.dropout.rng.rstate = self.pdict['__theano_mrg_rstate__']
            for (su2, su1) in zip(self.dropout.rng.state_updates, self.pdict['__theano_mrg_state_updates__'][0]):
                su2[0].set_value(su1[0].get_value())

