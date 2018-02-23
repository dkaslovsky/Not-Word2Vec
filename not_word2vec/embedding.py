""" Extremely initial implementation """

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from not_word2vec.count import WordCounter


class Embedding(WordCounter):

    def __init__(self, ngram_len, dim):
        super(Embedding, self).__init__(ngram_len)
        self.dim = dim

        # populated by fit
        self.vocab_ = None
        self.inv_vocab_ = None
        self.vocab_len_ = None
        self.U_ = None

    def fit(self, docs):

        unigram_counts, ngram_counts = self.count(docs)

        self.vocab_ = {word: i for i, word in enumerate(sorted(unigram_counts.iterkeys()))}
        self.inv_vocab_ = {v: k for k, v in self.vocab_.iteritems()}
        self.vocab_len_ = len(self.vocab_)

        P = self.pmi_matrix(ngram_counts, unigram_counts)
        self.U_ = self.embed(P, self.dim)

        return self

    def search(self, search_key, k=3):
        if isinstance(search_key, np.ndarray):
            if len(search_key) != self.U_.shape[1]:
                raise ValueError('search_key vector must be of shape (1, %i)' % self.U_.shape[1])
            vec = search_key
        elif isinstance(search_key, basestring):
            try:
                vec = self.U_[self.vocab_[search_key], :]
            except KeyError:
                raise ValueError('search_key must be in documents on which %s was fit'
                                 % self.__class__.__name__)
        else:
            raise TypeError('input type must be string or ndarray')

        dist = np.dot(self.U_, vec)
        idx = np.argpartition(-dist, k)[:k]
        return [self.inv_vocab_[i] for i in idx]

    def pmi_matrix(self, ngram_counts, unigram_counts):

        # compute row, column indices for sparse matrix
        # TODO: self.to_index(sorted=True) ?
        row_idx, col_idx = zip(*[self.to_index(key, sort=True) for key in ngram_counts.iterkeys()])

        # construct "adjacency" matrix of (log) joint probability pairs from skipgrams
        # TODO: check on definition as to whether this should be symmetric
        # TODO: if symmetric, divide by 2?
        joint_vals = np.array(ngram_counts.values())
        joint_vals = np.log(joint_vals / float(sum(joint_vals)))
        M_joint = csr_matrix((joint_vals, (row_idx, col_idx)), shape=(self.vocab_len_, self.vocab_len_))

        # construct "adjacency" matrix of (log) product of corresponding (independent) probabilities
        indep_vals = np.array([unigram_counts[i1] * unigram_counts[i2] for i1, i2 in ngram_counts.iterkeys()])
        indep_vals = np.log(indep_vals / float(sum(unigram_counts.itervalues()) ** 2))
        M_indep = csr_matrix((indep_vals, (row_idx, col_idx)), shape=(self.vocab_len_, self.vocab_len_))

        return M_joint - M_indep

    # TODO: if we want sort=False for PMI calculation, this serves little purpose and should be removed
    def to_index(self, key, sort=False):
        if isinstance(key, tuple):
            i1, i2 = self.vocab_[key[0]], self.vocab_[key[1]]
            if sort:
                i1, i2 = sorted([i1, i2])
            return i1, i2
        else:
            return self.vocab_[key]

    @staticmethod
    def embed(P, dim):
        U, _, _ = svds(P, k=dim)
        return U
