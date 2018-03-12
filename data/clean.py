import re

from sklearn.feature_extraction import stop_words


# words to be ignored
STOP_WORDS = stop_words.ENGLISH_STOP_WORDS
# characters to be removed from beginning and end of word
CHARS_TO_STRIP = '(){}[]!#.,?*\"\''
# retain only lower case words, allowing for apostrophes
REGEX = re.compile('^[a-z]+$')


def clean_documents(docs):
    """
    Clean documents and convert each to a list
    :param docs: iterable of string documents
    """
    return (_clean(doc) for doc in docs)


def _clean(doc):
    words = doc.split()
    words = _standardize(words)
    words = _remove_stop_words(words)
    words = _regex_filter(words)
    return words


def _regex_filter(word_list):
    return filter(REGEX.search, word_list)


def _remove_stop_words(word_list):
    return filter(lambda x: x not in STOP_WORDS, word_list)


def _standardize(word_list):
    return (word.strip(CHARS_TO_STRIP).replace('\'', '').lower()
            for word in word_list)
