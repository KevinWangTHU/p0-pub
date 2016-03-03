# Generate data to train autoencoders.
# The output has the same format as proc.py, but each doc contains only one sentence in a story, and the highlight is exactly the same.

#wdict["%%unk%%"] = 0
#wdict["%%para_end%%"] = 1
#wdict["%%doc_end%%"] = 2

# TODO: Check if the output contains "\.".
# TODO: Use the same format for proc.py

import nltk
import re
import os
import copy
import numpy as np

import gflags

gflags.DEFINE_string('root_path', './cnn/stories', 'directory containing all stories', short_name='p')
gflags.DEFINE_string('dump_prefix', './100k3', 'prefix of dict/data to be dumped', short_name='d')
gflags.DEFINE_integer('max_vocab', 100000, 'Max vocabulary size (including markers added by this script)', short_name='mv')
gflags.DEFINE_integer('max_tokens_per_sentence', None, '', short_name='mtps')
gflags.DEFINE_integer('max_paragraphs_per_document', None, '', short_name='mppd')

flags = gflags.FLAGS


def process_text(file_path):
    text = []
    with open(file_path) as fin:
        #
        lines = fin.readlines()
        lines = [l.strip() for l in lines]
        lines = [l for l in lines if len(l) > 0]
        #
        signs = re.compile('[\+\-\.\,\%]') # For digit removal
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        for l in lines:
            raw_sents = sent_tokenizer.tokenize(l.decode('utf-8'))
            print >>sys.stderr, raw_sents # FIXME
            sents = []
            for sent in raw_sents:
                tokens = nltk.tokenize.casual_tokenize(sent, preserve_case=False)
                tokens = [unicode(w) if not signs.sub('', w).isdigit() else u'0' \
                          for w in tokens][:flags.max_tokens_per_sentence]
                sents.append(tokens)

            text.append(sents)

    return text[:flags.max_paragraphs_per_document]


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
        n_sents = [[wdict[t] if t in wdict else 0 # 0<-%%unk%%
                    for t in sent] for sent in para] + [[1]] # 1<-%%para_end%%
        ndoc.append(n_sents)
    ndoc.append([[2]]) # 2<-%%doc_end%%
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
    
    # List stories in root_path
    files = [join(root_path, f) for f in os.listdir(root_path) \
             if isfile(join(root_path, f)) and splitext(join(root_path, f))[1] == '.story']
    
    # Tokenize stories & init dict
    stories = []
    wdict = {}
    len_files = len(files)
    n_empty = 0
    for i, story in enumerate(files):
        print >>sys.stderr, "\r%d/%d stories loaded" % (i, len_files),
        text = process_text(story)
        if len(text) == 0:
            n_empty += 1
            continue
        update_wdict(text, wdict)
        for paragraph in text:
            for sentence in paragraph:
                stories.append([[sentence]], [[copy.deepcopy(sentence)]])
    print >>sys.stderr, "\nStories loaded; %d empty stories excluded." % n_empty
    print >>sys.stderr, "%d sentences loaded." % len(stories)
    
    # Remove infrequent words & add markers
    word_freqs = wdict.items()
    word_freqs.sort(key=lambda x: -x[1])
    print >>sys.stderr, "%d words of %d (w/o markers) included" % (flags.max_vocab - 3, len(wdict))
    word_freqs = [(x, i + 3) for i, (x, _) in enumerate(word_freqs)][:flags.max_vocab - 3]
    wdict = dict(word_freqs)
    wdict["%%unk%%"] = 0
    wdict["%%para_end%%"] = 1
    wdict["%%doc_end%%"] = 2
    n_vocab = len(wdict)

    # Token->ID
    nstories = []
    for text, highlight in stories:
        nstories.append((update_doc(text, wdict), update_doc(highlight, wdict)))
    stories = nstories 
    
    #
    print >>sys.stderr, "document updated"
    # print wdict, stories[0]

    # 
    with open(dump_prefix + '.dict', 'wb') as fd:
        cPickle.dump(wdict, fd)
    with open(dump_prefix + '.train', 'wb') as ft:
        cPickle.dump(stories, ft)

if __name__ == '__main__':
    main()
