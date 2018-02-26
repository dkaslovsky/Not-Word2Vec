import itertools as it
import types
from collections import Counter


# TODO: protect against single doc - ensure list of lists

class WordCounter(object):

    def __init__(self, window_len):
        self.window_len = window_len
        self.skipgram_len = 2 * window_len + 1

    def count(self, docs):
        if isinstance(docs, types.GeneratorType):
            # use list instead of itertools.tee since we will
            # consume the entire generator before using it again
            docs = list(docs)
        return self.count_unigrams(docs), self.count_skipgram_pairs(docs)

    @staticmethod
    def count_unigrams(docs):
        return Counter(it.chain.from_iterable(docs))

    def count_skipgram_pairs(self, docs):
        counter = Counter()
        for doc in docs:
            skipgrams = self.skipgram_generator(self.pad_document(doc))
            pairs = [self.skipgram_to_pairs(skipgram) for skipgram in skipgrams]
            counter.update(it.chain.from_iterable(pairs))
        return counter

    def skipgram_to_pairs(self, skipgram):
        target_word = skipgram[self.window_len]
        context_words = it.chain(it.islice(skipgram, 0, self.window_len),
                                 it.islice(skipgram, self.window_len + 1, None))
        return [(target_word, context_word) for context_word in context_words if context_word is not None]

    def skipgram_generator(self, iterable):
        return it.izip(*[it.islice(seq, i, None) for i, seq in enumerate(it.tee(iterable, self.skipgram_len))])

    def pad_document(self, doc):
        return it.chain(it.repeat(None, self.window_len),
                     doc,
                     it.repeat(None, self.window_len))
