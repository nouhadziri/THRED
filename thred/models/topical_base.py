import codecs
import re

import tensorflow as tf

from ..util import vocab


def initialize_vocabulary(hparams):
    _create_vocabulary(hparams.vocab_file, hparams.topic_vocab_file, hparams.train_data, hparams.vocab_size)

    vocab_table = vocab.create_vocab_dict(hparams.vocab_file)
    topic_vocab_table = vocab.create_vocab_dict(hparams.topic_vocab_file)

    for w in topic_vocab_table:
        topic_vocab_table[w] = vocab_table[w]

    return vocab_table, topic_vocab_table


def _create_vocabulary(vocab_path, topic_vocab_path, data_path, max_vocabulary_size, normalize_digits=False):
    """A modified version of vocab.create_vocabulary
    """

    if tf.gfile.Exists(vocab_path) and tf.gfile.Exists(topic_vocab_path):
        return

    print("Creating vocabulary files from data %s" % data_path)
    dialog_vocab, topic_vocab = {}, {}

    def normalize(word):
        if normalize_digits:
            if re.match(r'[\-+]?\d+(\.\d+)?', word):
                return '<number>'

        return word

    with codecs.getreader('utf-8')(
            tf.gfile.GFile(data_path, mode="rb")) as f:
        counter = 0
        for line in f:
            counter += 1
            if counter % 100000 == 0:
                print("  processing line %d" % counter)

            delimited_line = line.split("\t")
            topic_tokens = delimited_line[-1].strip().split()
            dialog_tokens = " ".join([delimited_line[i].strip() for i in range(len(delimited_line) - 1)]).split()

            for word in dialog_tokens:
                word = normalize(word)

                if word in dialog_vocab:
                    dialog_vocab[word] += 1
                else:
                    dialog_vocab[word] = 1

            for word in topic_tokens:
                word = normalize(word)

                if word in topic_vocab:
                    topic_vocab[word] += 1
                else:
                    topic_vocab[word] = 1

    for word in topic_vocab:
        if word in dialog_vocab:
            topic_vocab[word] += dialog_vocab[word]

    topic_vocab_list = sorted(topic_vocab, key=topic_vocab.get, reverse=True)
    with codecs.getwriter('utf-8')(
            tf.gfile.GFile(topic_vocab_path, mode="wb")) as topic_vocab_file:
        for w in topic_vocab_list:
            topic_vocab_file.write(w + "\n")

    for reserved_word in vocab.RESERVED_WORDS:
        if reserved_word in dialog_vocab:
            dialog_vocab.pop(reserved_word)

    dialog_vocab_list = vocab.RESERVED_WORDS + sorted(dialog_vocab, key=dialog_vocab.get, reverse=True)

    if len(dialog_vocab_list) > max_vocabulary_size:
        dialog_vocab_list = dialog_vocab_list[:max_vocabulary_size]

    for word in topic_vocab:
        if word in dialog_vocab_list and word not in vocab.RESERVED_WORDS:
            dialog_vocab_list.remove(word)

    with codecs.getwriter('utf-8')(
            tf.gfile.GFile(vocab_path, mode="wb")) as vocab_file:
        for w in dialog_vocab_list:
            vocab_file.write(w + "\n")

        for w in topic_vocab_list:
            vocab_file.write(w + "\n")

    print("Topic vocabulary with {} words created".format(len(topic_vocab_list)))
    print("Vocabulary with {} words created".format(len(topic_vocab_list)))

    del topic_vocab
    del dialog_vocab
