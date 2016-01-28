import cPickle
import operator
import numpy as np
import theano

def concat(l):
    return reduce(operator.add, l)

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
                        all_mask_list.shape ~         same.
                        doc_sent_pos:                 as required by SentEncoder.forward
    """
    all_sent_list = []
    all_mask_list = []
    sum_sent_len = 0
    doc_sent_pos = np.zeros((max_doc_len, n_docs), dtype=np.int64)
    #
    sentences_annotated.sort(key=operator.itemgetter(0))
    for cur_batch_s in xrange(0, len(sentences_annotated), n_sent_batch):
        cur_sents = sentences_annotated[cur_batch_s: cur_batch_s + n_sent_batch]
        cur_batch_size = len(cur_sents)
        # all_sent_list
        max_sent_len = max([s[0] for s in cur_sents])
        sent_batch = np.zeros((max_sent_len + 1, cur_batch_size), dtype=np.int64)
        mask_batch = np.zeros((max_sent_len + 1, cur_batch_size), dtype=np.float32)
        #
        for j, s in enumerate(cur_sents):
            mask_batch[:s[0], j] = 1.
            for k, tok in enumerate(s[1]):
                sent_batch[k, j] = tok
        #
        all_sent_list.append(sent_batch)
        all_mask_list.append(mask_batch)
        sum_sent_len += max_sent_len + 1
        # doc_sent_pos
        for in_batch_pos, (_, tokens, doc_id, sent_id) in enumerate(cur_sents):
            pos = (sum_sent_len - 2) * n_sent_batch + in_batch_pos 
            doc_sent_pos[sent_id, doc_id] = pos
    #
    all_sent_list = np.concatenate(all_sent_list, axis=1)
    all_mask_list = np.concatenate(all_mask_list, axis=1)
    return all_sent_list, doc_sent_pos


def load_data(flags):
    """
    @return:    [((concated_sent, concated_mask, doc_sent_pos), 
                  (hl_sent_data, hl_sent_mask, hl_doc_mask)) for each batch]
    """
    docs = cPickle.load(flags['train_data'])

    # Sort wrt number of sentences
    n_docs = []
    for doc, highlight in docs:
        n_sent = sum([len(para) for para in doc])
        n_docs.append((n_sent, doc, highlight))
    n_docs.sort(key=lambda x: x[0])
    n_docs = [x[1:] for x in n_docs]

    # Build batches
    batches = []
    n_doc_batch = flags['n_doc_batch']
    n_sent_batch = flags['n_sent_batch']
    for i in xrange(0, len(docs), n_doc_batch):
        cur_docs = docs[i: i+n_doc_batch]

        # remove paragraph structure that is not yet utilized 
        cur_docs = [(concat(p), concat(h)) for p, h in cur_docs]
        # (TODO: add paragraph seperator token)
        
        # Document
        # - doc_mask
        cur_doc_lens = [len(p) for p, _ in cur_docs]
        max_doc_len = max(cur_doc_lens)
        doc_mask = np.zeros((max_doc_len, len(cur_docs)), dtype=np.float32)
        for j in len(cur_docs):
            doc_mask[:cur_doc_lens[j], j] = 1

        # - List of word-level batches
        cur_docs_annotated = [[(len(sent), sent, d_id, s_id) 
                               for s_id, sent in enumerate(doc)]
                              for d_id, doc in enumerate(cur_docs)]
        all_sent_annotated = concat(cur_docs_annotated)
        concated_sent, concated_mask, doc_sent_pos = build_sent_batch(all_sent_annotated, n_sent_batch, 
                                                                      len(cur_docs), max_doc_len)
        
        # Highlight
        # - mask
        cur_hl_lens = [len(h) for _, h in cur_docs]
        hl_doc_mask = np.zeros((max(cur_hl_lens), len(cur_docs)), dtype=np.float32)
        for j in len(cur_docs):
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
        
        batches.append(((concated_sent, concated_mask, doc_sent_pos),
                        (hl_sent_data, hl_sent_mask, hl_doc_mask)))

    return batches
