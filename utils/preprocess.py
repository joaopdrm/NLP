import re
import unicodedata

import nltk
from nltk import corpus
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer, WordNetLemmatizer


def preprocess(text: str, stemming=False, stopwords=False):
    """
    Preprocess will remove all special and alphanumerics characters from the text
    using portuguese stopwords.

    Parameters
    ----------
    text: str
        data that will be clean.
    stemming: False
        reduces the words to its radical
    stopwords: False
        remove too short words

    Returns
    -------
    text:
        Processed words.
    """
    text = text.lower()

    nfkd_form = unicodedata.normalize("NFKD", text)
    text = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

    regex = re.compile("<[^<>]+>")
    text = re.sub(regex, " ", text)

    regex = re.compile("(http|https)://[^\s]*")
    text = re.sub(regex, "<url>", text)

    regex = re.compile("[^\s]+@[^\s]+")
    text = re.sub(regex, "<email>", text)

    regex = re.compile("[^A-Za-z0-9]+")
    text = re.sub(regex, " ", text)

    regex = re.compile("[0-9]+")
    text = re.sub(regex, "<numero>", text)

    text = " ".join(text.split())

    words = text.split()

    words = words[0:200]

    if stopwords:
        words = text.split()
        words = [w for w in words if not w in nltk.corpus.stopwords.words("portuguese")]
        text = " ".join(words)

    if stemming:
        stemmer_method = RSLPStemmer()
        words = [stemmer_method.stem(w) for w in words]
        text = " ".join(words)

    words = text.split()
    words = [w for w in words if len(w) > 1 if len(w) > 1 or w == "0"]
    text = " ".join(words)

    return text
