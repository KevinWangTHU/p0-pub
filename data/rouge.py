import sumy.evaluation.rouge as rouge_
from sumy.models.dom import ObjectDocumentModel, Paragraph, Sentence
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import cached_property

class Sentence_(Sentence):
    def __init__(self, tokens):
        Sentence.__init__(self, "", Tokenizer('english'))
        self.tokens = tokens
    @cached_property
    def words(self):
        return self.tokens

def _rouge_l(s, r):
    try:
        return rouge_.rouge_l_sentence_level([s], [r])
    except ZeroDivisionError:
        return 0.0

def _rouge_1(s, r):
    if len(s.words) < 1 or len(r.words) < 1:
        return 0.0
    return rouge_.rouge_1([s], [r])

def _rouge_2(s, r):
    if len(s.words) < 2 or len(r.words) < 2:
        return 0.0
    return rouge_.rouge_2([s], [r])
    
def rouge(typ, sent, ref):
    import sys
    sent = Sentence_([unicode(w) for w in sent])
    ref = Sentence_([unicode(w) for w in ref])
    if typ == 'l':
        return _rouge_l(sent, ref)
    elif typ == '1p2':
        return _rouge_1(sent, ref) + _rouge_2(sent, ref)
    elif typ == '1':
        return _rouge_1(sent, ref)
    elif typ == '2':
        return _rouge_2(sent, ref)
    else:
        raise "Invalid type"

