""" Extremely initial implementation """

import numpy as np

from collections import Counter
from itertools import chain, combinations, islice, izip, tee
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import svds

from data import fetch_newsgroups_data, get_tokenized_docs



class NormalizedCounter(Counter):

    def normalize(self):
        total = float(sum(self.itervalues()))
        for key in self:
            self[key] /= total


class WordEmbedding(object):

    def __init__(self, ngram_len):
        self.ngram_len = ngram_len
        # populated by fit
        self.docs = None
        self.vocab_ = None
        self.inv_vocab_ = None
        self.unigram_prob_ = None
        self.ngram_prob_ = None
        self.U_ = None

    def fit(self, docs):
        # TODO: tee vs list, cache
        # https://stackoverflow.com/questions/21315207/deep-copying-a-generator-in-python/21315536#21315536
        # https://stackoverflow.com/questions/19503455/caching-a-generator
        self.docs = list(docs)
        self.unigram_prob_ = self.compute_unigram_probability()
        self.ngram_prob_ = self.compute_ngram_probability()
        # TODO: hashing trick?
        self.vocab_ = {word: i for i, word in enumerate(sorted(self.unigram_prob_.iterkeys()))}
        self.inv_vocab_ = {v: k for k, v in self.vocab_.iteritems()}
        return self

    def compute_unigram_probability(self):
        counts = self.count_unigrams()
        counts.normalize()
        return counts

    def compute_ngram_probability(self):
        counts = self.count_ngrams()
        counts.normalize()
        return counts

    def count_unigrams(self):
        return NormalizedCounter(chain.from_iterable(self.docs))

    def count_ngrams(self):
        counter = NormalizedCounter()
        for doc in self.docs:
            for ngram in self.ngram_generator(doc, self.ngram_len):
                pairs = combinations(np.unique(ngram), 2)
                counter.update(pairs)
        return counter

    # TODO: need to use adjustable size moving window?
    @staticmethod
    def ngram_generator(iterable, ngram_len):
        return izip(*[islice(seq, i, None) for i, seq in enumerate(tee(iterable, ngram_len))])

    def to_matrix(self):
        n = len(self.vocab_)
        S = dok_matrix((n, n), dtype=np.float32)
        for pair, prob in self.ngram_prob_.iteritems():
            S[self.to_index(pair)] = prob
        S = S + S.T
        return S

    def to_array(self):
        arr = np.empty((len(self.vocab_), 1))
        for word, prob in self.unigram_prob_.iteritems():
            arr[self.to_index(word)] = prob
        return arr

    def to_index(self, key):
        if isinstance(key, tuple):
            return self.vocab_[key[0]], self.vocab_[key[1]]
        else:
            return self.vocab_[key]

    def pmi(self):
        N = self.to_matrix()
        U = self.to_array()
        # TODO: make this sparse by masking U with indicator of N
        U_U_inv = dok_matrix(1.0 / np.dot(U, U.T))
        # TODO: should take a log here
        return N.multiply(U_U_inv)

    def embed(self, k):
        pmi = self.pmi()
        pmi.data = np.log(pmi.data)
        U, _, _ = svds(pmi, k=k)
        self.U_ = U

    def search(self, search_key):
        if isinstance(search_key, np.ndarray):
            # TODO: check lengths
            vec = search_key
        elif isinstance(search_key, basestring):
            try:
                vec = self.U_[self.vocab_[search_key], :]
            except KeyError:
                raise ValueError('search_key must be in documents on which %s was fit' % self.__class__.__name__)
        else:
            raise TypeError('Input type must be string or ndarray')
        # TODO: https://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
        # idx = np.argmax(np.dot(self.U_, vec))
        # return self.inv_vocab_[idx]
        idx = np.argsort(-np.dot(self.U_, vec))
        return [self.inv_vocab_[i] for i in idx[:3]]


if __name__ == '__main__':

    ngram_size = 5  # probably too small
    dim = 50

    data = fetch_newsgroups_data()
    docs = get_tokenized_docs(data[:100])

    we = WordEmbedding(ngram_size)
    we = we.fit(docs)
    we.embed(dim)

    for word in we.vocab_.iterkeys():
        print '%s -- %s' % (word, we.search(word))
