import numpy as np
import pandas as pd
import re

from collections import Counter
from itertools import chain, combinations, islice, izip, tee
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import stop_words as _stop_words


def fetch_data(stop_words, subset='all', clean=True):
    remove = ('headers', 'footers', 'quotes') if clean else ()
    newsgroups_docs = fetch_20newsgroups(subset=subset, remove=remove)
    return chain.from_iterable(clean_words(doc.split(), stop_words)
                               for doc in newsgroups_docs.data[:10])


def clean_words(word_iter, stop_words):
    chars_to_strip = '(){}[]!#.,?*\"\''
    regex = re.compile('^[a-z]+$')

    words = [word.strip(chars_to_strip).replace('\'', '').lower() for word in word_iter]
    words = filter(lambda x: x not in stop_words, words)
    words = filter(regex.search, words)
    return words


def unigram_counts(data):
    return Counter(data)


def ngram_generator(iterable, n):
    return izip(*[islice(seq, i, None) for i, seq in enumerate(tee(iterable, n))])


def ngram_counts(data, n):
    counter = Counter()
    for ngram in ngram_generator(data, n):
        pairs = combinations(np.unique(ngram), 2)
        counter.update(pairs)
    return counter


# TODO: more efficient implementation?
def normalize(counter):
    total = float(sum(counter.itervalues()))
    return {k : v / total for k, v in counter.iteritems()}


# brutal initial implementation - this should be a sparse matrix, etc
def pmi(ngram_probs, word_probs, symmetric=False):
    vocab = sorted(word_probs.keys())
    adj = pd.DataFrame(0, columns=vocab, index=vocab)
    for pair, count in ngram_probs.iteritems():
        adj.loc[pair] = count
    if symmetric:
        adj += adj.T
    word_prob_df = pd.Series(word_probs).to_frame().loc[vocab]
    adj /= np.dot(word_prob_df, word_prob_df.T)
    adj = np.log(adj[adj > 0])
    return adj


def get_edges(adj, thresh=0):
    edges = []
    for idx, row in adj.iterrows():
        pairs = [(idx, col) for col in row[row > thresh].index]
        edges.extend(pairs)
    return edges


if __name__ == '__main__':

    ngram_size = 5
    stop_words = _stop_words.ENGLISH_STOP_WORDS

    data = fetch_data(stop_words)
    data_iter1, data_iter2 = tee(data, 2)

    word_probabilities = normalize(unigram_counts(data_iter1))
    ngram_probabilities = normalize(ngram_counts(data_iter2, ngram_size))

    adj = pmi(ngram_probabilities, word_probabilities, symmetric=True)
