import nltk
import re
import os
import numpy as np


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
                          for w in tokens]
                sents.append(tokens)

            if is_highlight:
                highlight.append(sents) 
                is_highlight = False
            else:
                text.append(sents)

    return text, highlight 


def update_wdict(doc, wdict):
    for para in doc:
        for tokens in para:
            for w in tokens:
                if not w in wdict:
                    t = len(wdict)
                    wdict[w] = t


def update_doc(doc, wdict):
    ndoc = []
    for para in doc:
        n_sents = [[wdict[t] for t in sent] for sent in para]
        ndoc.append(n_sents)
    return ndoc


def main(root_path, dump_prefix):
    import sys
    from os.path import isfile, join, splitext
    from operator import itemgetter
    import cPickle
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

    print >>sys.stderr, "%d words included" % len(wdict)

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
    import sys
    main(sys.argv[1], sys.argv[2])
