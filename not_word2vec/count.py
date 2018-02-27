import itertools as it
import types
from collections import Counter


class WordCounter(object):

    def __init__(self, window_len):
        """

        :param window_len: int length of one side of skipgram window
        """
        self.window_len = window_len
        self.skipgram_len = 2 * window_len + 1

    def count(self, docs):
        """
        Compute unigram and skipgram-pair counts
        :param docs: iterable of lists where each list contains strings
          in order of appearance in a document
        """
        if isinstance(docs, types.GeneratorType):
            # use list instead of itertools.tee since we will
            # consume the entire generator before using it again
            docs = list(docs)
        return self.count_unigrams(docs), self.count_skipgram_pairs(docs)

    @staticmethod
    def count_unigrams(docs):
        """
        Compute word counts
        :param docs: iterable of lists where each list contains strings
          in order of appearance in a document
        """
        return Counter(it.chain.from_iterable(docs))

    def count_skipgram_pairs(self, docs):
        """
        Compute counts of pairs of the form (word i, word j)
        :param docs: iterable of lists where each list contains strings
          in order of appearance in a document
        """
        counter = Counter()
        for doc in docs:
            skipgrams = self._skipgram_generator(self._pad_document(doc))
            pairs = [self._skipgram_to_pairs(skipgram) for skipgram in skipgrams]
            counter.update(it.chain.from_iterable(pairs))
        return counter

    def _skipgram_to_pairs(self, skipgram):
        """
        Construct list of (target, context) tuples where target is the center position
         of the skipgram
        :param skipgram: iterable/generator of strings of length 2 * self.window_len + 1
        """
        target_word = skipgram[self.window_len]
        context_words = it.chain(it.islice(skipgram, 0, self.window_len),
                                 it.islice(skipgram, self.window_len + 1, None))
        return [(target_word, context_word) for context_word in context_words if context_word is not None]

    def _skipgram_generator(self, doc):
        """
        Construct generator of skipgrams
        :param doc: list of strings
        """
        return it.izip(*[it.islice(seq, i, None) for i, seq in enumerate(it.tee(doc, self.skipgram_len))])

    def _pad_document(self, doc):
        """
        Add self.window_len None values to the beginning and end of doc
        :param doc: list of strings
        """
        return it.chain(it.repeat(None, self.window_len),
                     doc,
                     it.repeat(None, self.window_len))
