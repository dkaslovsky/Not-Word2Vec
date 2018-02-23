import types
from collections import Counter
from itertools import chain, combinations, islice, izip, tee

import numpy as np


class WordCounter(object):

    def __init__(self, ngram_len):
        self.ngram_len = ngram_len

    def count(self, docs):
        if isinstance(docs, types.GeneratorType):
            # use list instead of itertools.tee since we will
            # consume the entire generator before using it again
            docs = list(docs)
        return self.count_unigrams(docs), self.count_ngrams(docs)

    @staticmethod
    def count_unigrams(docs):
        return Counter(chain.from_iterable(docs))

    # TODO: symmetric or not?
    def count_ngrams(self, docs):
        counter = Counter()
        for doc in docs:
            for ngram in self.ngram_generator(doc):
                # np.unique returns results in sorted order
                pairs = combinations(np.unique(ngram), 2)
                counter.update(pairs)
        return counter

    # TODO: need to use adjustable size moving window?
    def ngram_generator(self, iterable):
        return izip(*[islice(seq, i, None) for i, seq in enumerate(tee(iterable, self.ngram_len))])
