import codecs
import re
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

UNK, UNK_ID = "<UNK>", 0
SOS, SOS_ID = "<S>", 1
EOS, EOS_ID = "</S>", 2
SEP, SEP_ID = "<SEP>", 3

RESERVED_WORDS = [UNK, SOS, EOS, SEP]


def load_vocab(vocab_file):
    vocab = []

    if tf.gfile.Exists(vocab_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
            vocab_size = 0
            for word in f:
                vocab_size += 1
                vocab.append(word.strip())
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_file)

    return vocab, vocab_size


def create_vocab_table(vocab_file):
    """Creates vocab tables for vocab_file."""
    return lookup_ops.index_table_from_file(vocab_file, default_value=UNK_ID)


def create_rev_vocab_table(vocab_file):
    return lookup_ops.index_to_string_table_from_file(vocab_file, default_value=UNK)


def create_vocab_dict(vocab_file, start_index=0):
    """Creates vocab tables for vocab_file."""
    if tf.gfile.Exists(vocab_file):

        vocab_dict = defaultdict(lambda: UNK_ID)

        with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
            for id, word in enumerate(f):
                w = word.strip()
                vocab_dict[w] = id + start_index

        return vocab_dict
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_file)


def save_vocab_dict(vocab_file, vocab_dict):
    with codecs.getwriter("utf-8")(tf.gfile.GFile(vocab_file, mode="wb")) as vocab_file:
        for w in sorted(vocab_dict, key=vocab_dict.get):
            vocab_file.write(w + "\n")


# _WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, normalize_digits=False):
    """Create vocabulary file (if it does not exist yet) from data file.
    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.
    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """

    if not tf.gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with codecs.getreader("utf-8")(tf.gfile.GFile(data_path, "rb")) as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                tokens = line.replace(u"\u00A0", " ").split()
                for w in tokens:
                    word = _DIGIT_RE.sub("0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

            for reserved_word in RESERVED_WORDS:
                if reserved_word in vocab:
                    vocab.pop(reserved_word)
            vocab_list = RESERVED_WORDS + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with codecs.getwriter("utf-8")(tf.gfile.GFile(vocabulary_path, mode="wb")) as vocab_file:
                vocab_size = len(vocab_list)
                for i, w in enumerate(vocab_list):
                    vocab_file.write(w + ("\n" if i < vocab_size - 1 else ""))
