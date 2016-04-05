#wdict["%%unk%%"] = 0
#wdict["%%para_end%%"] = 1
#wdict["%%doc_end%%"] = 2

import nltk
import re
import os
import numpy as np
from nltk.tokenize import StanfordTokenizer
import rouge

import gflags
import sys

reload(sys)
sys.setdefaultencoding('utf8')

# ${root_path}/stories & ${root_path}/highlights must exist
gflags.DEFINE_string('root_path', './cnn', 'directory containing all stories', short_name='p')
gflags.DEFINE_string('dump_prefix', './100r3', 'prefix of dict/data to be dumped', short_name='d')
gflags.DEFINE_integer('max_vocab', 100000, 'Max vocabulary size (including markers added by this script)', short_name='mv')
gflags.DEFINE_integer('max_tokens_per_sentence', None, '', short_name='mtps')
gflags.DEFINE_integer('max_paragraphs_per_document', None, '', short_name='mppd')
#gflags.DEFINE_string('dict', None, '', short_name='dic')
gflags.DEFINE_string('dict', '/home/wzy/glove/glove/vectors.txt', '', short_name='dic')
gflags.DEFINE_integer('n_embed', 100, '', short_name='ne')
gflags.DEFINE_integer('n_tokenize_batch', 500, '', short_name='ntb')
gflags.DEFINE_enum('score_type', '1p2', ['1', '2', '1p2', 'l'], 'type of matching score (rouge-*)', short_name='st')
gflags.DEFINE_bool('debug', False, 'use small dataset for debugging')

flags = gflags.FLAGS

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


def update_doc(doc, wdict, hlt=False):
    ndoc = []
    for para in doc:
        n_sents = [[wdict[t] if t in wdict else 0 # 0<-%%unk%%
                    for t in sent] for sent in para] + [[1]] # 1<-%%para_end%%
        if hlt:
            sent = reduce(lambda a, b: a+b, n_sents[:-1])
            n_sents = [sent, [1]]
        ndoc.append(n_sents)
    ndoc.append([[2]]) # 2<-%%doc_end%%
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


def match_score(txt, hlt):
    """
    return: np.array((n_sents, n_hlts), f32)
    """
    txt = reduce(lambda a, b: a+b, txt)
    hlt = reduce(lambda a, b: a+b, hlt)
    ret = np.zeros((len(txt), len(hlt)), dtype='float32')
    for i, t in enumerate(txt):
        for j, h in enumerate(hlt):
            ret[i, j] = rouge.rouge(flags.score_type, t, h)
    return ret


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
    story_root_path = join(root_path, 'stories')
    stries = [join(story_root_path, f) for f in os.listdir(story_root_path) \
               if isfile(join(story_root_path, f)) and splitext(join(story_root_path, f))[1] == '.story']
    hlt_path = join(root_path, 'highlights')
    hlts = [join(hlt_path, re.sub('.story$', '.highlight', f)) for f in os.listdir(story_root_path) \
            if isfile(join(story_root_path, f)) and splitext(join(story_root_path, f))[1] == '.story']

    if flags.debug:
        stries = stries[:50]
        hlts = hlts[:50]

    # Tokenize stories & init dict
    wdict = {}
    len_files = len(stries)
    n_empty = 0
    stories = []
    for i in xrange(0, len_files, flags.n_tokenize_batch):
        print >>sys.stderr, "\r%d/%d stories loaded" % (i, len_files),
        texts = batch_tokenize(stries[i: i+flags.n_tokenize_batch])
        highlights = batch_tokenize(hlts[i: i+flags.n_tokenize_batch])
        for (txt, hlt) in zip(texts, highlights):
            if len(txt) == 0 or len(hlt) == 0:
                n_empty += 1
                continue
            text = process_lines(txt)
            highlight = process_lines(hlt)
            update_wdict(text, wdict)
            update_wdict(highlight, wdict)
            stories.append((text, highlight))
    print >>sys.stderr, "\nStories loaded; %d empty stories excluded." % n_empty

    # Remove infrequent words & add markers
    word_freqs = wdict.items()
    word_freqs.sort(key=lambda x: -x[1])
    if flags.dict:
        pre_embed, pre_dict = load_vecs(flags.dict)
        word_freqs = filter(lambda x: x[0] in pre_dict, word_freqs)
        word_freqs = [("<unk>", 0), ("<para_end>", 0), ("<doc_end>", 0)] + word_freqs
        word_freqs = [(x, i) for i, (x, _) in enumerate(word_freqs)][:flags.max_vocab]
        shin_embed = np.array([pre_embed[pre_dict[w]] for w, _ in word_freqs])
    else:
        print >>sys.stderr, "%d words of %d (w/o markers) included" % (flags.max_vocab - 3, len(wdict))
        word_freqs = [(x, i + 3) for i, (x, _) in enumerate(word_freqs)][:flags.max_vocab - 3]
        word_freqs = [("<unk>", 0), ("%%para_end%%", 1), ("%%doc_end%%", 2)] + word_freqs
    #
    wdict = dict(word_freqs)
    n_vocab = len(wdict)

    # Token->ID; calculate rouge
    nstories = []
    for i, (text, highlight) in enumerate(stories):
        print >>sys.stderr, "\r%d/%d stories scored" % (i, len_files - n_empty),
        text_, highlight_ = update_doc(text, wdict), update_doc(highlight, wdict, hlt=True)
        score_ = match_score(text_, highlight_)
        nstories.append(((text_, highlight_), score_))
    print >>sys.stderr, "\nDone." 
    stories = nstories 

    if flags.debug:
        import IPython
        IPython.embed()

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
