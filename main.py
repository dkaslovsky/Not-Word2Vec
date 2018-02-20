""" Extremely initial implementation """

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
                               for doc in newsgroups_docs.data)


def clean_words(word_iter, stop_words):
    chars_to_strip = '(){}[]!#.,?*\"\''
    regex = re.compile('^[a-z]+$')

    words = [word.strip(chars_to_strip).replace('\'', '').lower() for word in word_iter]
    words = filter(lambda x: x not in stop_words, words)
    words = filter(regex.search, words)
    return words


def unigram_counts(data):
    return Counter(data)


# TODO: need to use adjustable size moving window?
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


def embed(adj, dim):
    data = adj.fillna(0).values
    if dim > min(data.shape):
        raise ValueError('Must specify embedding dimension <= min(adj.shape)')
    U, _, _ = np.linalg.svd(data, full_matrices=False)
    return pd.DataFrame(U[:, :dim], index=adj.index)


def search(embedding, vec):
    idx = np.argmax(np.dot(embedding, vec))
    return embedding.index[idx]


def get_edges(adj, thresh=0):
    edges = []
    for idx, row in adj.iterrows():
        pairs = [(idx, col) for col in row[row > thresh].index]
        edges.extend(pairs)
    return edges


if __name__ == '__main__':

    ngram_size = 7
    stop_words = _stop_words.ENGLISH_STOP_WORDS

    data = fetch_data(stop_words)
    data_iter1, data_iter2 = tee(data, 2)

    word_probabilities = normalize(unigram_counts(data_iter1))
    ngram_probabilities = normalize(ngram_counts(data_iter2, ngram_size))

    adj = pmi(ngram_probabilities, word_probabilities, symmetric=True)

    embedding = embed(adj, 25)

    for word in embedding.index:
        print '%s -- %s' % (word, search(embedding, embedding.loc[word]))
