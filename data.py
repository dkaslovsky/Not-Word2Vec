import re

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import stop_words


STOP_WORDS = stop_words.ENGLISH_STOP_WORDS
CHARS_TO_STRIP = '(){}[]!#.,?*\"\''


def fetch_newsgroups_data(subset='all', clean=True):
    remove = ('headers', 'footers', 'quotes') if clean else ()
    newsgroups_docs = fetch_20newsgroups(subset=subset, remove=remove)
    return newsgroups_docs.data


def tokenize(word_iter):
    # strip chars, remove apostrophes, cast to lowercase
    words = (word.strip(CHARS_TO_STRIP).replace('\'', '').lower() for word in word_iter)
    # remove stop words
    words = filter(lambda x: x not in STOP_WORDS, words)
    # only allow words made up of lowercase letters
    regex = re.compile('^[a-z]+$')
    words = filter(regex.search, words)
    return words


def get_tokenized_docs(docs):
    return (tokenize(doc.split()) for doc in docs)



class Tokenizer(object):

    def __init__(self):
        self.regex = re.compile('^[a-z]+$')
        self.chars_to_strip = CHARS_TO_STRIP
        self.stop_words = STOP_WORDS

    def tokenize(self, docs):
        """

        :param docs: iterable of documents (strings)
        :return:
        """
        return (self._process_doc(doc.split()) for doc in docs)

    def _process_doc(self, word_list):
        words = self._clean(word_list)
        words = self._remove_stop_words(words)
        words = self._regex_filter(words)
        return words

    def _regex_filter(self, word_list):
        return filter(self.regex.search, word_list)

    def _remove_stop_words(self, word_list):
        return filter(lambda x: x not in self.stop_words, word_list)

    def _clean(self, word_list):
        return (word.strip(self.chars_to_strip).replace('\'', '').lower()
                for word in word_list)
