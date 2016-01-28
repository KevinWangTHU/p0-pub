import nltk
import re
import os
import numpy as np

import gflags

gflags.DEFINE_string('root_path', '.', '', short_name='p')
gflags.DEFINE_string('dump_prefix', './test', '', short_name='d')
gflags.DEFINE_integer('max_vocab', -1, '', short_name='mv')
gflags.DEFINE_integer('max_tok_per_sent', -1, '', short_name='mtps')
gflags.DEFINE_integer('max_para_per_doc', -1, '', short_name='mppd')

flags = gflags.FLAGS


def process_text(file_path):
    text = []
    highlight = []
    with open(file_path) as fin:
        #
        lines = fin.readlines()
        lines = [l.strip() for l in lines]
        lines = [l for l in lines if len(l) > 0]
        #
        signs = re.compile('[\+\-\.\,\%]')
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        is_highlight = False
        for l in lines:
            if l == '@highlight':
                is_highlight = True
                continue
            #
            raw_sents = sent_tokenizer.tokenize(l)
            sents = []
            for sent in raw_sents:
                tokens = nltk.tokenize.casual_tokenize(sent, preserve_case=False)
                tokens = [unicode(w) if not signs.sub('', w).isdigit() else u'0' \
                          for w in tokens][:flags.max_tok_per_sent]
                sents.append(tokens)

            if is_highlight:
                highlight.append(sents) 
                is_highlight = False
            else:
                text.append(sents)

    return text[:flags.max_para_per_doc], highlight 


def update_wdict(doc, wdict):
    for para in doc:
        for tokens in para:
            for w in tokens:
                if not w in wdict:
                    wdict[w] = 0
                wdict[w] += 1


def update_doc(doc, wdict):
    ndoc = []
    for para in doc:
        n_sents = [[wdict[t] if t in wdict else 0
                    for t in sent] for sent in para] + [[1]]
        ndoc.append(n_sents)
    return ndoc


def main():
    import sys
    from os.path import isfile, join, splitext
    from operator import itemgetter
    import cPickle
    #
    sys.argv = flags(sys.argv)
    root_path = flags.root_path
    dump_prefix = flags.dump_prefix
    #
    files = [join(root_path, f) for f in os.listdir(root_path) \
             if isfile(join(root_path, f)) and splitext(join(root_path, f))[1] == '.story']
    #
    stories = []
    wdict = {}
    for story in files:
        text, highlight = process_text(story)
        update_wdict(text, wdict)
        update_wdict(highlight, wdict)
        stories.append((text, highlight))
    #
    word_freqs = wdict.items()
    word_freqs.sort(key=lambda x: -x[1])
    print >>sys.stderr, "%d words of %d included" % (flags.max_vocab + 2, len(wdict))
    word_freqs = [(x, i + 2) for i, (x, _) in enumerate(word_freqs)][:flags.max_vocab]
    wdict = dict(word_freqs)
    wdict["%%unk%%"] = 0
    wdict["%%para_end%%"] = 1
    n_vocab = len(wdict)
    #
    nstories = []
    for text, highlight in stories:
        nstories.append((update_doc(text, wdict), update_doc(highlight, wdict)))
    stories = nstories 
    #
    print >>sys.stderr, "document updated"
    #
    print wdict, stories[0]
    with open(dump_prefix + '.dict', 'wb') as fd:
        cPickle.dump(wdict, fd)
    with open(dump_prefix + '.train', 'wb') as ft:
        cPickle.dump(stories, ft)

if __name__ == '__main__':
    main()
