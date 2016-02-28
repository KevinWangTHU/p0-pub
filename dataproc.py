import cPickle
import operator
import numpy as np
import theano

from util import *


def build_sent_batch(sentences_annotated, n_sent_batch, n_docs, max_doc_len):
    """
    @param sentences\: [(sentence_length, token_list, doc_id, sentence_id) 
          _annotated    for all sentences in a batch]
    @return:           [all_sent_list, all_mask_list, doc_sent_pos], where
                        all_sent_list, all_mask_list: WordEncoder inputs ((index, mask)) for this 
                                                      batch;
                                                      sentences are seperated by a 0 in mask.
                        all_sent_list.shape ~         (sum(max_sent_len_in_batch: for each batch), 
                                                       n_sent_batch)
                        all_mask_list.shape ~         [same]
                        doc_sent_pos:                 as required by SentEncoder.forward
    """
    all_sent_list = []
    all_mask_list = []
    sum_sent_len = 0
    doc_sent_pos = np.zeros((max_doc_len, n_docs), dtype=np.int64)
    #
    for cur_batch_s in xrange(0, len(sentences_annotated), n_sent_batch):
        cur_sents = sentences_annotated[cur_batch_s: cur_batch_s + n_sent_batch]
        # all_sent_list
        max_sent_len = max([s[0] for s in cur_sents])
        sent_batch = np.zeros((max_sent_len + 1, n_sent_batch), dtype=np.int64)
        mask_batch = np.zeros((max_sent_len + 1, n_sent_batch), dtype=np.float32)
        #
        for j, s in enumerate(cur_sents):
            for k, tok in enumerate(s[1]):
                sent_batch[k, j] = tok
                mask_batch[k, j] = 1.
        #
        all_sent_list.append(sent_batch)
        all_mask_list.append(mask_batch)
        sum_sent_len += max_sent_len + 1
        # doc_sent_pos
        for in_batch_pos, (_, tokens, doc_id, sent_id) in enumerate(cur_sents):
            pos = (sum_sent_len - 2) * n_sent_batch + in_batch_pos 
            doc_sent_pos[sent_id, doc_id] = pos
    #
    all_sent_list = np.concatenate(all_sent_list, axis=0)
    all_mask_list = np.concatenate(all_mask_list, axis=0)
    return all_sent_list, all_mask_list, doc_sent_pos


def build_input(docs, flags):
    """
    @return:    [(id, 
                  (concated_sent, concated_mask, doc_sent_pos, doc_mask, 
                   hl_sent_data, hl_sent_mask, hl_doc_mask),
                  original highlights ([[[int] for highlight in doc] for doc in batch]))
                 for each batch]
                if flags['simplernn'] == True, 
                    all sentences will be concatenated in a document, and
                    concated_sent[:, i] corresponds to the ith document.
    """

    n_docs = []
    if not flags['simplernn']:
        # Sort documents by number of sentences
        for doc, highlight in docs:
            n_sent = sum([len(para) for para in doc])
            n_docs.append((n_sent, doc, highlight))
        n_docs.sort(key=lambda x: x[0])
        n_docs = [x[1:] for x in n_docs]
    else:
        # Concatenate all paragraphs & sentences (keeping the list hierarchy), 
        # and sort wrt number of words.
        for doc, highlight in docs:
            highlight = [[concat(concat(highlight))]]
            if not flags['__ae__']:
                doc = [[concat(concat(doc))]]
            else:
                import copy
                doc = copy.deepcopy(highlight)
                highlight = [[list(reversed(highlight[0][0]))]]
            #
            n_word = len(doc[0][0])
            n_docs.append((n_word, doc, highlight))
        n_docs.sort(key=lambda x: x[0])
        n_docs = [x[1:] for x in n_docs]            

    # TODO: Implement reverse_output

    # Build batches
    batches = []
    n_doc_batch = flags['n_doc_batch']
    n_sent_batch = flags['n_sent_batch']
    for i in xrange(0, len(n_docs), n_doc_batch):
        cur_docs = n_docs[i: i+n_doc_batch]

        # remove paragraph structure that is not yet utilized 
        cur_docs = [(concat(p), concat(h)) for p, h in cur_docs]
        
        # Document
        # - doc_mask
        cur_doc_lens = [len(p) for p, _ in cur_docs]
        max_doc_len = max(cur_doc_lens)
        doc_mask = np.zeros((max_doc_len, len(cur_docs)), dtype=np.float32)
        for j in xrange(len(cur_docs)):
            doc_mask[:cur_doc_lens[j], j] = 1.

        # - List of word-level batches
        cur_docs_annotated = [[(len(sent), sent, d_id, s_id) 
                               for s_id, sent in enumerate(doc)]
                              for d_id, (doc, _) in enumerate(cur_docs)]
        all_sent_annotated = concat(cur_docs_annotated)
        if not flags['simplernn']:
            # Speedup
            all_sent_annotated.sort(key=operator.itemgetter(0))
            concated_sent, concated_mask, doc_sent_pos = build_sent_batch(
                all_sent_annotated, n_sent_batch, len(cur_docs), max_doc_len)
        else:
            concated_sent, concated_mask, doc_sent_pos = build_sent_batch(
                all_sent_annotated, len(cur_docs), len(cur_docs), max_doc_len)
        
        # Highlight
        # - mask
        cur_hl_lens = [len(h) for _, h in cur_docs]
        hl_doc_mask = np.zeros((max(cur_hl_lens), len(cur_docs)), dtype=np.float32)
        for j in xrange(len(cur_docs)):
            hl_doc_mask[:cur_hl_lens[j], j] = 1

        # - data & sent_mask
        max_hl_sent_len = reduce(max, [max([len(s) for s in hl]) for _, hl in cur_docs])
        hl_sent_data = np.zeros((max(cur_hl_lens), max_hl_sent_len, len(cur_docs)), dtype=np.int64)
        hl_sent_mask = np.zeros((max(cur_hl_lens), max_hl_sent_len, len(cur_docs)), dtype=np.float32)
        for doc_id, (_, hl) in enumerate(cur_docs):
            for hl_id, hl_sent in enumerate(hl):
                for tok_id, token in enumerate(hl_sent):
                    hl_sent_data[hl_id, tok_id, doc_id] = token
                    hl_sent_mask[hl_id, tok_id, doc_id] = 1.
        
        train_input = (concated_sent, concated_mask, doc_sent_pos, doc_mask, 
                       hl_sent_data, hl_sent_mask, hl_doc_mask)
        highlights = [h for d, h in cur_docs]
        batches.append((i, train_input, highlights))

    log_info({'type': 'data', 'value': 'data loaded'})
    return batches


def load_data(flags):
    all_docs = None
    with open(flags['train_data'] + '.train') as fin:
        all_docs = cPickle.load(fin)
    np.random.seed(7297)
    np.random.shuffle(all_docs)
    split = len(all_docs) * 4 / 5
    return build_input(all_docs[:split], flags), \
           build_input(all_docs[split:], flags)


def load_test_data(flags):
    with open(flags['train_data'] + '.train') as fin:
        all_docs = cPickle.load(fin)
    np.random.seed(7297)
    np.random.shuffle(all_docs)
    split = len(all_docs) * 4 / 5
    ret = build_input(all_docs[-100:], flags)
    np.random.shuffle(ret)
    return ret
