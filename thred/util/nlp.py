import collections
import re
import codecs
from os.path import join

import emot
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from . import misc, fs
from .twokenize import tokenize as tweet_tokenize
from .twitter_nlp_emoticons import Emoticon_RE


def _read_emots():
    emots = set()
    with codecs.getreader('utf-8')(open(join(fs.get_current_dir(__file__), 'emots.txt'), 'rb')) as e:
        for line in e:
            emots.add(line.strip())

    return emots


UNCOMMON_EMOTICONS = _read_emots()


class TaggedWord(
    collections.namedtuple("TaggedWord", ("index", "term", "lemma", "pos", "ner"))):

    def __repr__(self) -> str:
        if self.ner:
            return "{term}@{index}/{pos}~{ner}".format(term=self.term,
                                                       index=self.index,
                                                       pos=self.pos,
                                                       ner=self.ner)
        else:
            return "{term}@{index}/{pos}".format(term=self.term,
                                                 index=self.index,
                                                 pos=self.pos)


class NLPToolkit:
    def __init__(self, pipeline=None):
        self._nlp = spacy.load('en_core_web_lg')

        if pipeline is not None:
            for name in pipeline:
                component = self._nlp.create_pipe(name)
                self._nlp.add_pipe(component)

    def sent_tokenize(self, text):
        doc = self._nlp(text)
        return [sent.string.strip() for sent in doc.sents]

    def annotate(self, text):
        doc = self._nlp(text)
        return NLPToolkit._parse_doc(doc)

    def tokenize(self, text):
        doc = self._nlp(text)
        return [w.text for w in doc]

    @staticmethod
    def _parse_doc(doc):
        sentences = []
        for sent in doc.sents:
            tagged_words = []
            for token in sent.doc:
                tagged_words.append(TaggedWord(index=token.i,
                                               term=NLPToolkit.replace_treebank_standards(token.text),
                                               lemma=token.lemma_,
                                               pos=token.pos_ if doc.is_tagged else None,
                                               ner=token.ent_type_))
            sentences.append(tagged_words)
        return sentences

    @staticmethod
    def replace_treebank_standards(token):
        if token in ('``', "''"):
            return '"'
        elif token == '`':
            return "'"
        elif token == '-LRB-':
            return "("
        elif token == '-RRB-':
            return ")"
        elif token == '-LCB-':
            return "{"
        elif token == '-RCB-':
            return "}"
        elif token == '-LSB-':
            return "["
        elif token == '-RSB-':
            return "]"
        else:
            return token

    @staticmethod
    def is_stopword(word):
        return word in STOP_WORDS


def normalize_entities(sentences, entities=None, decapitalize=True):
    if entities is None:
        entities = {'PERSON': '<person>',
                    'URL': '<url>',
                    'NUMBER': '<number>',
                    'PERCENT': '<number>',
                    'DURATION': '<number>',
                    'MONEY': '<number>',
                    'DATE': '<time>',
                    'TIME': '<time>'}

    if decapitalize:
        def get_str(t):
            return t.lower()
    else:
        def get_str(t):
            return t

    normalized_text = []

    for tagged_words in sentences:
        for tagged_word in tagged_words:
            if tagged_word.ner in entities:
                named_entity = entities[tagged_word.ner]
            else:
                named_entity = ''

            if named_entity:
                if tagged_word.ner in ('DATE', 'MONEY', 'NUMBER', 'TIME', 'DURATION', 'PERCENT') and not re.match(r'[\-+]?\d+(\.\d+)?', tagged_word.term):
                    normalized_text.append(get_str(tagged_word.term))
                elif not normalized_text or normalized_text[-1] != named_entity:
                    normalized_text.append(named_entity)
            else:
                normalized_text.append(get_str(tagged_word.term))

    return normalized_text


def strip_emojis_and_emoticons(text):
    return _strip_emoticons(_strip_emojis(text))


def _strip_emojis(text):
    emojis = set([emoji['value'] for emoji in emot.emoji(text)])

    normalized = text
    for emoji in emojis:
        normalized = normalized.replace(emoji, '')

    return normalized


def _strip_emoticons(text):
    global UNCOMMON_EMOTICONS

    tokens = tweet_tokenize(text)

    emoticons = set()
    for token in tokens:
        for em in emot.emoticons(token):
            emoticon = em['value']
            if emoticon in ('(', ')', ':') or emoticon != token:
                continue
            emoticons.add(emoticon)

        if Emoticon_RE.match(token) or token in (':*(',):
            emoticons.add(token)

    for em in UNCOMMON_EMOTICONS:
        if em in text:
            emoticons.add(em)

    normalized = text
    for emoticon in emoticons:
        if re.match(r'^[a-zA-Z0-9]+$', emoticon.lower()):
            continue

        if re.match(r'^[a-zA-Z0-9].*', emoticon):
            if re.match(r'.*\b{}.*'.format(misc.escRegex(emoticon)), normalized):
                normalized = normalized.replace(emoticon, '')
        elif re.match(r'.*[a-zA-Z0-9]$', emoticon):
            if re.match(r'.*{}\b.*'.format(misc.escRegex(emoticon)), normalized):
                normalized = normalized.replace(emoticon, '')
        else:
            if re.match(r'.*\s{}.*'.format(misc.escRegex(emoticon)), normalized) or \
                    re.match(r'.*{}\s.*'.format(misc.escRegex(emoticon)), normalized) or \
                    re.match(r'^{}$'.format(misc.escRegex(emoticon)), normalized):
                normalized = normalized.replace(emoticon, '')

    normalized = re.sub(r'(^|\s)([;:8=][\-^]\s+[><}{)(|/*x$#&3D0OoPpc\[\]])(.*)', r'\1\3', normalized)
    normalized = re.sub(r'(.*)([;:8=][\-^]\s+[><}{)(|/*x$#&3D0OoPpc\[\]])(\s|$)', r'\1\3', normalized)

    return normalized


if __name__ == "__main__":
    nlp_toolkit = NLPToolkit()

    print(strip_emojis_and_emoticons(
        'Clearly the media has replaced stormy with Comey 😂😂😂😂 :/ :) :-( ;) ಠಿ_ಠ'))
