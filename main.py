import time

from data import fetch_newsgroups_data
from not_word2vec import Embedding, Tokenizer


if __name__ == '__main__':

    n_docs = 1000

    skipgram_window_size = 4
    embedding_dimension = 200

    data = fetch_newsgroups_data()
    docs = Tokenizer().tokenize(data[:n_docs])

    e = Embedding(skipgram_window_size, embedding_dimension)

    t_start = time.time()
    e = e.fit(docs)
    t_end = time.time()
    print 'Computed %i dimensional embedding of vocabulary of size %i in %i second(s)' \
          % (embedding_dimension, e.vocab_len_, (t_end - t_start))

    search_key = 'sportscenter'
    print 'Words similar to \'%s\' %s: ' % (search_key, e.search(search_key, k=10))

    word1, word2 = ('sportscenter', 'news')
    print '%s - %s = %s' % (word1, word2, e.subtract(word1, word2))