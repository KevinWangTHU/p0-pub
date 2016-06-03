#wdict["%%unk%%"] = 0
#wdict["%%para_end%%"] = 1
#wdict["%%doc_end%%"] = 2

import nltk
import re
import os
import numpy as np
from nltk.tokenize import StanfordTokenizer

import gflags
import sys
import IPython
from sumy.nlp.tokenizers import Tokenizer
from sumy.evaluation.rouge import rouge_n, rouge_l_sentence_level
from sumy.models.dom import Sentence

reload(sys)
sys.setdefaultencoding('utf8')

# ${root_path}/stories & ${root_path}/highlights must exist
# gflags.DEFINE_string('root_path', './cnn', 'directory containing all stories', short_name='p')
gflags.DEFINE_string('root_path', './dm', 'directory containing all stories and picks and highlights', short_name='p')
gflags.DEFINE_string('dump_prefix', './100k5', 'prefix of dict/data to be dumped', short_name='d')
gflags.DEFINE_integer('max_vocab', 100000, 'Max vocabulary size (including markers added by this script)', short_name='mv')
gflags.DEFINE_integer('max_tokens_per_sentence', None, '', short_name='mtps')
gflags.DEFINE_integer('max_paragraphs_per_document', None, '', short_name='mppd')
#gflags.DEFINE_string('dict', None, '', short_name='dic')
gflags.DEFINE_string('dict', '/home/wzy/glove/glove/vectors.txt', '', short_name='dic')
gflags.DEFINE_integer('n_embed', 100, '', short_name='ne')
gflags.DEFINE_integer('n_tokenize_batch', 500, '', short_name='ntb')

flags = gflags.FLAGS

import operator
def concat(l):
    return reduce(operator.add, l)

def process_text_legacy(file_path):
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
            sents = []
            for sent in raw_sents:
                tokens = nltk.tokenize.casual_tokenize(sent, preserve_case=False)
                tokens = [unicode(w) if not signs.sub('', w).isdigit() else u'0' \
                          for w in tokens][:flags.max_tokens_per_sentence]
                sents.append(tokens)
            text.append(sents)

    return text[:flags.max_paragraphs_per_document]


tokenizer = StanfordTokenizer(path_to_jar='/home/wzy/stanford_parser/src/stanford-parser.jar')

# Batched tokenization with Stanford tokenizer (Java startup is expensive)
def batch_tokenize(file_paths):
    to_be_tokenized = u""
    for file_path in file_paths:
        with open(file_path) as fin:
            lines = fin.readlines()
            lines = [l.strip() for l in lines]
            lines = [l.decode() for l in lines if len(l) > 0]
            lines = u'<para_end>\n'.join(lines)
        to_be_tokenized += lines + '<doc_end>\n'

    global tokenizer
    tokens = tokenizer.tokenize(to_be_tokenized)
    tokens = [tok.lower() for tok in tokens]
    docs = u' '.join(tokens).split('<doc_end>')[:-1]
    assert len(docs) == len(file_paths)
    doc_lines = [doc.split('<para_end>') for doc in docs]
    return doc_lines


def batch_digit(file_paths):
    digits = []
    for file_path in file_paths:
        with open(file_path) as fin:
            lines = fin.readlines()
            lines = [l.strip() for l in lines]
            lines = [ord(l[0]) - 48 for l in lines]
            digits.append(lines)
    assert len(digits) == len(file_paths)
    return digits

def process_lines(lines):
    lines = filter(lambda x: len(x) > 0, lines)
    text = []
    signs = re.compile('[\+\-\.\,\%]') # Remove symbols in numbers
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for l in lines:
        raw_sents = sent_tokenizer.tokenize(l.decode('utf-8'))
        sents = []
        for sent in raw_sents:
            tokens = sent.split(u' ')
            tokens = filter(lambda x: len(x) > 0, tokens)
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
                    for t in sent] for sent in para] # + [[1]] # 1<-%%para_end%%
        ndoc.append(n_sents)
    # ndoc.append([[2]]) # 2<-%%doc_end%%
    return ndoc


def load_vecs(path):
    with open(path) as fin:
        lines = fin.readlines()
        ret = []
        wordmap = []
        for i, line in enumerate(lines):
            line = line.split(u' ')
            word = line[0]
            vec = [float(f) for f in line[1:]]
            ret.append(vec)
            wordmap.append((word, i))
    return np.array(ret, dtype='float32'), dict(wordmap)

R1_T = 0.37
R2_T = 0.17
SENT_T = 3
def _pick(docs, refs):
    "@return: a 0/1 vector"

    outlabels = [0] * len(docs)
    for ref in refs:
        r1 = []
        r2 = []
        for i, sent in enumerate(docs):
            r1.append((rouge_n([sent], [ref], 1), i))
            r2.append((rouge_n([sent], [ref], 2), i))
        r1.sort(key=lambda x:x[0], reverse=True)
        r2.sort(key=lambda x:x[0], reverse=True)
        for idx, t in enumerate(r1):
            if t[0] < R1_T or idx > SENT_T:
                break
            outlabels[t[1]] = 1
        for idx, t in enumerate(r2):
            if t[0] < R2_T or idx > SENT_T:
                break
            outlabels[t[1]] = 1
    return outlabels

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
    story_root_path = root_path
    stries = [join(story_root_path, f) for f in os.listdir(story_root_path) \
               if isfile(join(story_root_path, f)) and splitext(join(story_root_path, f))[1] == '.story']
    hlt_path = root_path
    hlts = [join(hlt_path, re.sub('.story$', '.highlight', f)) for f in os.listdir(story_root_path) \
            if isfile(join(story_root_path, f)) and splitext(join(story_root_path, f))[1] == '.story']
    pk_path = root_path
    pks = [join(pk_path, re.sub('.story$', '.pick', f)) for f in os.listdir(story_root_path) \
            if isfile(join(story_root_path, f)) and splitext(join(story_root_path, f))[1] == '.story']

    # Tokenize stories & init dict
    wdict = {}
    len_files = len(stries)
    n_empty = 0
    stories = []
    tokenizer = Tokenizer("english")
    s1 = 0
    s2 = 0

    for i in xrange(0, len_files, flags.n_tokenize_batch):
        print >>sys.stderr, "\r%d/%d stories loaded" % (i, len_files),
        texts = batch_tokenize(stries[i: i+flags.n_tokenize_batch])
        highlights = batch_tokenize(hlts[i: i+flags.n_tokenize_batch])
#         picks = batch_digit(pks[i: i+flags.n_tokenize_batch])
        for (txt, hl) in zip(texts, highlights):
            text = process_lines(txt)
            highlight = process_lines(hl)

            original_text = [" ".join(w) for w in concat(text)]
            original_highlight = [" ".join(w) for w in concat(highlight)]
            text_sents = [Sentence(s, tokenizer) for s in original_text]
            highlight_sents = [Sentence(s, tokenizer) for s in original_highlight]
            try:
                pick_sents = _pick(text_sents, highlight_sents)
                update_wdict(text, wdict)
                stories.append((text, pick_sents, original_text, original_highlight))
                s1 += len(pick_sents)
                s2 += sum(pick_sents)
            except ZeroDivisionError:
                n_empty += 1

    print >>sys.stderr, "\nStories loaded; %d empty stories excluded." % n_empty
    print >>sys.stderr, s1, s2, s2/(s1+.0)

    # Remove infrequent words & add markers
    word_freqs = wdict.items()
    word_freqs.sort(key=lambda x: -x[1])
    if flags.dict:
        pre_embed, pre_dict = load_vecs(flags.dict)
        word_freqs = filter(lambda x: x[0] in pre_dict, word_freqs)
        # word_freqs = [("<unk>", 0), ("<para_end>", 0), ("<doc_end>", 0)] + word_freqs
        word_freqs = [("<unk>", 0)] + word_freqs
        word_freqs = [(x, i) for i, (x, _) in enumerate(word_freqs)][:flags.max_vocab]
        shin_embed = np.array([pre_embed[pre_dict[w]] for w, _ in word_freqs])
    else:
        print >>sys.stderr, "%d words of %d (w/o markers) included" % (flags.max_vocab - 3, len(wdict))
        # word_freqs = [(x, i + 3) for i, (x, _) in enumerate(word_freqs)][:flags.max_vocab - 3]
        # word_freqs = [("<unk>", 0), ("%%para_end%%", 1), ("%%doc_end%%", 2)] + word_freqs
        word_freqs = [(x, i + 1) for i, (x, _) in enumerate(word_freqs)][:flags.max_vocab - 3]
        word_freqs = [("<unk>", 0)] + word_freqs
    #
    wdict = dict(word_freqs)
    n_vocab = len(wdict)


    # Token->ID
    nstories = []
    for text, pick, original_text, original_highlight in stories:
        nstories.append((update_doc(text, wdict), pick, original_text, original_highlight))
    stories = nstories

    #
    print >>sys.stderr, "document updated"
    # print wdict, stories[0]

    #
    with open(dump_prefix + '.dict', 'wb') as fd:
        cPickle.dump(wdict, fd)
    with open(dump_prefix + '.train', 'wb') as ft:
        cPickle.dump(stories, ft)
    if flags.dict:
        with open(dump_prefix + '.embed', 'wb') as fe:
            cPickle.dump(shin_embed, fe)

if __name__ == '__main__':
    main()
