import unittest

from not_word2vec.embedding import Embedding


class TestEmbedding(unittest.TestCase):
    """
    Unit tests for Embedding
    """

    docs = ['a b c b e'.split(), 'f b a c a'.split()]

    def setUp(self):
        self.window_len = 1
        self.dim = 3
        self.e = Embedding(self.window_len, self.dim)

    def test_fit(self):
        self.e = self.e.fit(self.docs)
        self.assertItemsEqual(self.e.U_.shape, (5, self.dim))
        # TODO: test vocab

    def test_pmi(self):
        # TODO: make populate_vocab a separate function to avoid calling all of fit?
        self.e = self.e.fit(self.docs)
        unigram_counts, skipgram_counts = self.e.word_counter.count(self.docs)
        P = self.e._pmi_matrix(skipgram_counts, unigram_counts)
        self.assertItemsEqual(P.shape, (5, 5))
        # TODO: other tests here?
