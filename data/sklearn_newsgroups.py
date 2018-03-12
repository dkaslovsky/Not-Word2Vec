from sklearn.datasets import fetch_20newsgroups

from .clean import clean_documents


def fetch_newsgroups_data(n_docs=None, clean=True):
    remove = ('headers', 'footers', 'quotes')
    newsgroups_docs = fetch_20newsgroups(subset='all', remove=remove)
    data = newsgroups_docs.data
    if n_docs:
        data = data[:n_docs]
    if clean:
        return clean_documents(data)
    return data
