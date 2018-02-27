import re

from sklearn.feature_extraction import stop_words


STOP_WORDS = stop_words.ENGLISH_STOP_WORDS
CHARS_TO_STRIP = '(){}[]!#.,?*\"\''


class Tokenizer(object):

    def __init__(self):
        self.regex = re.compile('^[a-z]+$')
        self.chars_to_strip = CHARS_TO_STRIP
        self.stop_words = STOP_WORDS

    def tokenize(self, docs):
        """

        :param docs: iterable of string documents
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
