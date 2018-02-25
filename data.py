from sklearn.datasets import fetch_20newsgroups


def fetch_newsgroups_data(subset='all', clean=True):
    remove = ('headers', 'footers', 'quotes') if clean else ()
    newsgroups_docs = fetch_20newsgroups(subset=subset, remove=remove)
    return newsgroups_docs.data
