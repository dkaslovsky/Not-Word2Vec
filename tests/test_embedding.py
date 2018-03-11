import numpy as np
import unittest

from itertools import chain

from not_word2vec.embedding import Embedding
from not_word2vec.count import WordCounter


class TestEmbedding(unittest.TestCase):
    """
    Unit tests for Embedding
    """

    docs = ['a b c b e'.split(), 'f b a c a'.split()]
    vocab_len = len(np.unique(list(chain.from_iterable(docs))))

    def setUp(self):
        # set up Embedding
        self.window_len = 1
        self.dim = 3
        self.e = Embedding(self.window_len, self.dim)
        # set up unigram and skipgram counts from WordCounter for validation
        wc = WordCounter(self.window_len)
        unigram_counts, skipgram_counts = wc.count(self.docs)
        self.unigram_counts = unigram_counts
        self.skipgram_counts = skipgram_counts

    def test_set_vocab(self):
        self.e._set_vocab(self.unigram_counts)
        # vocab has correct length
        self.assertEqual(self.e.vocab_len_, self.vocab_len)
        # vocab contains expected keys
        self.assertListEqual(self.e.vocab_.keys(), self.unigram_counts.keys())
        # inverse vocab properly constructed
        self.assertDictEqual(self.e.inv_vocab_, {v:k for k, v in self.e.vocab_.iteritems()})

    def test_pmi(self):
        self.e._set_vocab(self.unigram_counts)
        P = self.e._pmi_matrix(self.skipgram_counts, self.unigram_counts)
        # test dimensions are correct
        self.assertItemsEqual(P.shape, (self.vocab_len, self.vocab_len))
        # test P is symmetric
        self.assertTrue(np.all(P.A == P.A.T))
        # test PMI(a, b) > 0 since they should co-occur in a skipgram
        self.assertTrue(P[self.e.vocab_['a'], self.e.vocab_['b']] > 0)
        # test PMI(e, f) = 0 (really would be -inf if not using sparse matrix)
        self.assertEqual(P[self.e.vocab_['e'], self.e.vocab_['f']], 0)
        # test for zero diagonal
        self.assertListEqual(list(P.diagonal()), [0] * self.vocab_len)

    def test_embed(self):
        self.e = self.e.embed(self.docs)
        # test e.U_ has expected dimensions
        self.assertItemsEqual(self.e.U_.shape, (self.vocab_len, self.dim))

    def test_to_vec(self):
        self.e = self.e.embed(self.docs)
        # test e.to_vec('a') corresponds to the vector embedding of 'a'
        self.assertTrue(np.allclose(self.e.to_vec('a'), self.e.U_[self.e.vocab_['a'], :]))
        # test e.to_vec of word not in vocab raises error
        with self.assertRaises(ValueError):
            self.e.to_vec('z')

    def test_search(self):
        self.e = self.e.embed(self.docs)
        # test searching for 'a' returns a word in the vocab
        self.assertTrue(self.e.search('a', 1)[0] in self.e.vocab_)
        # test searching for a vector returns a word in the vocab
        self.assertTrue(self.e.search(np.array([0, 1, 2]), 1)[0] in self.e.vocab_)
        # test searching for word not in vocab raises error
        with self.assertRaises(ValueError):
            self.e.search('z')
        # test searching for array with wrong dimension raises error
        with self.assertRaises(ValueError):
            self.e.search(np.array([0, 1, 2, 3]))
        # test searching for type other than string or array raises error
        with self.assertRaises(TypeError):
            self.e.search((0, 1, 2))

    def test_add(self):
        self.e = self.e.embed(self.docs)
        # test a + b is in the vocab
        self.assertTrue(self.e.add('a', 'b', 1)[0] in self.e.vocab_)

    def test_subtract(self):
        self.e = self.e.embed(self.docs)
        # test a - b is in the vocab
        self.assertTrue(self.e.subtract('a', 'b', 1)[0] in self.e.vocab_)
