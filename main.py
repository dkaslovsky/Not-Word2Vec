import time

from data import fetch_newsgroups_data
from not_word2vec import Embedding, Tokenizer


if __name__ == '__main__':

    n_docs = 100

    skipgram_window_size = 2  # corresponds to skipgram len = 5; probably too small
    embedding_dimension = 50

    data = fetch_newsgroups_data()
    docs = Tokenizer().tokenize(data[:n_docs])

    e = Embedding(skipgram_window_size, embedding_dimension)

    t_start = time.time()
    e = e.fit(docs)
    t_end = time.time()
    print 'Embedded vocabulary of size %i in %i second(s)' % (e.vocab_len_, (t_end - t_start))

    print e.search('sportscenter', 10)
