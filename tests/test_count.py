import unittest

from not_word2vec.count import WordCounter


class TestWordCounter(unittest.TestCase):
    """
    Unit tests for WordCounter
    """

    doc = 'a b c b e'.split()
    expected_unigram_counts = {'a': 1, 'b': 2, 'c': 1, 'e': 1}
    expected_pair_counts = {
        ('a', 'b'): 1,
        ('b', 'a'): 1,
        ('b', 'c'): 2,
        ('b', 'e'): 1,
        ('c', 'b'): 2,
        ('e', 'b'): 1
    }

    def setUp(self):
        self.window_len = 1
        self.wc = WordCounter(self.window_len)

    def test_pad_document(self):
        padded = self.wc._pad_document(self.doc)
        expected_padded = [None] + self.doc + [None]
        self.assertEqual(list(padded), expected_padded)

    def test_skipgram_generator(self):
        skipgrams = self.wc._skipgram_generator(self.doc)
        self.assertEqual(skipgrams.next(), ('a', 'b', 'c'))
        self.assertEqual(skipgrams.next(), ('b', 'c', 'b'))
        self.assertEqual(skipgrams.next(), ('c', 'b', 'e'))
        with self.assertRaises(StopIteration):
            skipgrams.next()

    def test_skipgram_to_pairs(self):
        pairs = self.wc._skipgram_to_pairs(self.doc[:3])
        expected_pairs = [('b', 'a'), ('b', 'c')]
        self.assertEqual(pairs, expected_pairs)

    def test_count_skipgram_pairs(self):
        pair_counts = self.wc.count_skipgram_pairs([self.doc])
        self.assertEqual(pair_counts, self.expected_pair_counts)

    def test_count_unigrams(self):
        unigram_counts = self.wc.count_unigrams([self.doc[:3], self.doc[3:]])
        self.assertEqual(unigram_counts, self.expected_unigram_counts)

    def test_count(self):
        unigram_counts, skipgram_counts = self.wc.count([self.doc, self.doc])
        expected_unigram_counts = {k: 2 * v for k, v in self.expected_unigram_counts.iteritems()}
        expected_pair_counts = {k: 2 * v for k, v in self.expected_pair_counts.iteritems()}
        self.assertEqual(unigram_counts, expected_unigram_counts)
        self.assertEqual(skipgram_counts, expected_pair_counts)
