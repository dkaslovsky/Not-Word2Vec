# Not-Word2Vec
This is not word2vec.

[![Build Status](https://travis-ci.org/dkaslovsky/Not-Word2Vec.svg?branch=master)](https://travis-ci.org/dkaslovsky/Not-Word2Vec)
[![Coverage Status](https://coveralls.io/repos/github/dkaslovsky/Not-Word2Vec/badge.svg?branch=master)](https://coveralls.io/github/dkaslovsky/Not-Word2Vec?branch=master)


Pointwise Mutual Information (PMI) implementation of word embedding algorithm as described by https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/

Implementation is "by-hand" -- avoids using standard NLP tools such as those found in `NLTK`, `gensim`, `scikit-learn`, because what fun would that be?

To run unit tests:
```
$ python -m unittest discover -v tests
```

See [example.ipynb](example.ipynb) for example embedding, word search, and analogies via word addition and word subtraction.
