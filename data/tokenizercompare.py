# -*- coding: utf-8 -*-

import os
from shutil import copy2 as copy
# from __future__ import absolute_import
# from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import nltk
import operator

def concat(l):
    return reduce(operator.add, l)

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def cpr(source):
    story_dir = "dm/"
    storyFiles = os.listdir(story_dir)
    for i, fileName in enumerate(storyFiles):
        if not fileName.endswith(".story"):
            continue
        aa=1
        sents1 = PlaintextParser.from_file(story_dir+fileName, Tokenizer("english")).document.sentences
        lines = [l for l in open(story_dir+fileName).readlines()]
        # lines = [l for l in lines if len(l) > 0]
        sents2 = sent_tokenizer.tokenize(concat(lines))
        for sent1, sent2 in zip(sents1, sents2):
            print "---"
            print sent1
            print sent2
        # print sents2
        print len(sents1), len(sents2)
        break

if __name__ == "__main__":
    cpr("dm")
