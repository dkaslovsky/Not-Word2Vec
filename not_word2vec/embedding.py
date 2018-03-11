import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from not_word2vec.count import WordCounter


class Embedding(object):

    def __init__(self, window_len, dim):
        """

        :param window_len: int length of one side of skipgram window
        :param dim: int dimension of space into which words will be embdedded
        """
        self.word_counter = WordCounter(window_len)
        self.dim = dim
        # populated by fit
        self.vocab_ = None
        self.inv_vocab_ = None
        self.vocab_len_ = None
        self.U_ = None

    def fit(self, docs):
        """
        Construct embedding for words contained in documents
        :param docs: generator or iterable of lists of strings
        """
        # get counts
        unigram_counts, skipgram_counts = self.word_counter.count(docs)
        # set vocabulary
        self._set_vocab(unigram_counts)
        # construct PMI matrix
        P = self._pmi_matrix(skipgram_counts, unigram_counts)
        # construct embedding
        self.U_ = self._embed(P, self.dim)
        return self

    def search(self, search_key, k=5):
        """
        Find the k closest words in the embedded space to search_key
        :param search_key: string in self.vocab_ or vector/array of shape (1, self.dim)
        :param k: int number of words to return
        """
        if isinstance(search_key, basestring):
            vec = self.to_vec(search_key)
        elif isinstance(search_key, np.ndarray):
            if len(search_key) != self.U_.shape[1]:
                raise ValueError('search_key vector must be of shape (1, %i)'
                                 % self.U_.shape[1])
            vec = search_key
        else:
            raise TypeError('input type must be string or ndarray')

        dist = np.dot(self.U_, vec)
        idx = np.argpartition(-dist, k)[:k]
        return [self.inv_vocab_[i] for i in idx]

    def to_vec(self, word):
        """
        Compute vector representation of word
        :param word: string
        """
        try:
            idx = self.vocab_[word]
        except KeyError:
            raise ValueError('%s not found in vocabulary' % word)
        return self.U_[idx, :]

    def add(self, word1, word2, k=5, return_vec=False):
        """
        Addition of word vectors
        :param word1: string in self.vocab_.keys()
        :param word2: string in self.vocab_.keys()
        :param k: number of search results to return
        :param return_vec: True if vector representation should be returned
         instead of word search results
        """
        vec = self.to_vec(word1) + self.to_vec(word2)
        if return_vec:
            return vec
        return self.search(vec, k=k)

    def subtract(self, word1, word2, k=5, return_vec=False):
        """
        Subtraction of word vectors
        :param word1: string in self.vocab_.keys()
        :param word2: string in self.vocab_.keys()
        :param k: number of search results to return
        :param return_vec: True if vector representation should be returned
         instead of word search results
        """
        vec = self.to_vec(word1) - self.to_vec(word2)
        if return_vec:
            return vec
        return self.search(vec, k=k)

    def _set_vocab(self, unigram_counts):
        """
        Construct mapping and inverse mapping of word : row index
        :param unigram_counts: dict mapping word i : observed count
        """
        self.vocab_ = {word: i for i, word in enumerate(sorted(unigram_counts.iterkeys()))}
        self.inv_vocab_ = {v: k for k, v in self.vocab_.iteritems()}
        self.vocab_len_ = len(self.vocab_)
        return

    def _pmi_matrix(self, skipgram_counts, unigram_counts):
        """
        Construct sparse PMI matrix where entry i, j = log(p(i,j)/p(i)p(j)), with
         p(i) = probability/frequency of word i
         p(i, j) = joint probability (co-occurrence frequency) of words i and j
        :param skipgram_counts: dict mapping (word i, word j): observed count
        :param unigram_counts: dict mapping word i : observed count
        """

        # compute row, column indices for sparse matrix
        row_idx, col_idx = zip(*[(self.vocab_[key[0]], self.vocab_[key[1]])
                                 for key in skipgram_counts.iterkeys()])

        # construct "adjacency" matrix of (log) joint probability
        #  pairs from skipgrams
        joint_vals = np.array(skipgram_counts.values())
        joint_vals = np.log(joint_vals / float(sum(joint_vals)))
        M_joint = csr_matrix((joint_vals, (row_idx, col_idx)),
                             shape=(self.vocab_len_, self.vocab_len_))

        # construct "adjacency" matrix of (log) product
        # of corresponding (independent) probabilities
        indep_vals = np.array([unigram_counts[key1] * unigram_counts[key2]
                               for key1, key2 in skipgram_counts.iterkeys()])
        indep_vals = np.log(indep_vals / float(sum(unigram_counts.itervalues()) ** 2))
        M_indep = csr_matrix((indep_vals, (row_idx, col_idx)),
                             shape=(self.vocab_len_, self.vocab_len_))

        return M_joint - M_indep

    @staticmethod
    def _embed(P, dim):
        """
        Compute embedding
        :param P: PMI (sparse) matrix
        :param dim: embedding dimension
        """
        U, _, _ = svds(P, k=dim)
        return U
