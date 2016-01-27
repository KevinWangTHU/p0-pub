import theano
import theano.tensor as T

def test_scan():
    x = T.fmatrix('x')
    ruler = T.fvector('ruler')
    def step(seqv, pre_result, *args):
        return pre_result + T.dot(seqv, ruler)
    ret, _ = theano.scan(fn=step,
                         sequences=x,
                         outputs_info=0.0,
                         non_sequences=ruler)
    return theano.function([x, ruler], ret[-1], name='f233')

def test_scan_2():
    def gen(x):
        ret, _ = theano.scan(fn=lambda p, x: (p+x, p+x+x),
                             non_sequences=x,
                             outputs_info=[T.zeros_like(x),None],
                             n_steps=5)
        return ret
    x = T.matrix('x')
    return theano.function([x], gen(x), name='f')

f = test_scan_2()
print f([[1.0], [2.0]])
