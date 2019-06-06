import string

from ..util.nlp import NLPToolkit

enhanced_punctuation = string.punctuation + \
                       u'\u2012' + u'\u2013' + u'\u2014' + u'\u2015' + u'\u2018' + u'\u2019' + u'\u201C' + u'\u201D' + \
                       u'\u2212' + u'\u2026'
translate_table = dict((ord(char), u'') for char in enhanced_punctuation)

contractions = {"'ll", "'ve", "'re", "n't", "doesn't", "don't", "i'm"}


def normalize(word, min_length=None):
    """
    converts terms in lower case, drops stop words and applies stemming using
    the PorterStemmer algorithm
    """

    term = word.lower()

    if NLPToolkit.is_stopword(term) or term in contractions:
        raise Warning("stopwords are not normalized here")

    if min_length is not None and len(word) < min_length:
        raise Warning("word is too short")

    term = term.translate(translate_table)

    if not term:
        raise Warning("after normalization word became empty")

    return term


def normalize_sequence(words, min_length=None):
    normalized = []
    for word in words:
        try:
            term = normalize(word, min_length)
            normalized.append(term)
        except Warning:
            continue

    return normalized
