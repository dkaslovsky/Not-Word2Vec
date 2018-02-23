import unittest

from not_word2vec.count import WordCounter


DOCS = [
    'the quick brown fox jumps',
    'over the lazy dog'
]

UNIGRAMS = {
    'the': 2, 'quick': 1,
    'brown': 1, 'fox': 1,
    'jumps': 1, 'over': 1,
    'lazy': 1, 'dog': 1,
}

NGRAMS = {
    ('brown', 'quick'): 2, ('brown', 'fox'): 2, ('lazy', 'the'): 2,
    ('dog', 'lazy'): 1, ('brown', 'the'): 1, ('dog', 'the'): 1,
    ('quick', 'the'): 1, ('fox', 'quick'): 1, ('over', 'the'): 1,
    ('brown', 'jumps'): 1, ('fox', 'jumps'): 1, ('lazy', 'over'): 1,
}


def docs_generator():
    for doc in DOCS:
        yield doc.split()


class TestWordCounter(unittest.TestCase):

    def setUp(self):
        ngram_len = 3
        self.wc = WordCounter(ngram_len)

    def test_count(self):
        docs = docs_generator()
        unigrams, ngrams = self.wc.count(docs)
        self.assertEqual(unigrams, UNIGRAMS)
        self.assertEqual(ngrams, NGRAMS)

    def test_ngram_generator(self):
        doc = DOCS[0].split()
        ngrams = self.wc.ngram_generator(doc)
        self.assertEqual(ngrams.next(), ('the', 'quick', 'brown'))
        self.assertEqual(ngrams.next(), ('quick', 'brown', 'fox'))
        self.assertEqual(ngrams.next(),('brown', 'fox', 'jumps'))
        with self.assertRaises(StopIteration):
            ngrams.next()
