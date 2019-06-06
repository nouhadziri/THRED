import codecs
import logging
from os import listdir, mkdir
from os.path import isdir, exists, join, abspath

import gensim
import yaml
from gensim import corpora

from . import analyzer
from ..util import fs
from ..util.misc import Stopwatch


class LDAArgs(dict):
    def __init__(self, params=None, *args, **kwargs):
        super(LDAArgs, self).__init__(*args, **kwargs)
        self.update(params)
        self.__dict__ = self

    def save(self, args_file):
        to_dump_dict = dict(self.__dict__)
        to_dump_dict['documents'] = abspath(to_dump_dict['documents'])

        with codecs.getwriter("utf-8")(open(args_file, "wb")) as f:
            yaml.dump(to_dump_dict, f, default_flow_style=False)

    @staticmethod
    def load(args_file):
        with codecs.getreader("utf-8")(open(args_file, "rb")) as f:
            params = yaml.load(f)

        return LDAArgs(params=params)


def iter_corpus(documents, min_length=None):
    if not exists(documents):
        raise ValueError('The documents data does not exist: {}'.format(documents))

    all_docs = []
    sw = Stopwatch()

    if isdir(documents):
        print('Documents stored as files in directory "{}"'.format(documents))

        files = listdir(documents)

        for i, f in enumerate(files):
            if not f.endswith('.txt'):
                continue

            file_path = join(documents, f)

            doc = []
            with codecs.getreader("utf-8")(open(file_path, 'rb')) as f:
                for line in f:
                    words = line.split()
                    for word in words:
                        try:
                            term = analyzer.normalize(word, min_length)
                            doc.append(term)
                        except Warning:
                            continue

            all_docs.append(doc)
            if i % 1000 == 0:
                sw.print('  {} of {} iterated'.format(i, len(files)))
    else:
        print('Documents stored in each line in file "{}"'.format(documents))
        with codecs.getreader("utf-8")(open(documents, 'rb')) as f:
            for i, line in enumerate(f):
                doc = analyzer.normalize_sequence(line.split(), min_length)

                if doc:
                    all_docs.append(doc)

                if i % 100000 == 0:
                    sw.print('  {} lines iterated'.format(i))

    sw.print('corpus built')
    return all_docs


def train(model_dir, args):
    if not exists(model_dir):
        mkdir(model_dir)

    corpus = iter_corpus(args.documents, args.min_length)
    dictionary = corpora.Dictionary(corpus)
    dictionary.filter_extremes(no_below=args.no_below)

    mm_corpus_file = join(model_dir, 'corpus.mm')

    if not exists(mm_corpus_file):
        print("corpus not found. Starting to build it...")

        class CorpusWrapper:

            def __init__(self, dictionary):
                self._dictionary = dictionary

            def __iter__(self):
                for tokens in corpus:
                    yield self._dictionary.doc2bow(tokens)

        gensim.corpora.MmCorpus.serialize(mm_corpus_file, CorpusWrapper(dictionary))

    mm_corpus = gensim.corpora.MmCorpus(mm_corpus_file)

    # generate LDA model
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    ldamodel = gensim.models.LdaMulticore(mm_corpus,
                                          id2word=dictionary,
                                          alpha='asymmetric', eta='auto',
                                          num_topics=args.num_topics,
                                          passes=args.passes,
                                          eval_every=args.eval_every,
                                          batch=True,
                                          chunksize=args.chunksize,
                                          iterations=args.iterations)
    print("Saving LDA model...")
    ldamodel.save(join(model_dir, 'LDA.model'))

    print("Saving words for topics...")
    with open(join(model_dir, 'TopicWords.txt'), 'w') as topic_file:
        for i in range(args.num_topics):
            topic_file.write('Topic #{}:\n\t'.format(i))
            topic_words_ids = [x[0] for x in ldamodel.get_topic_terms(i, topn=args.words_per_topic)]
            topic_file.write('\n\t'.join([dictionary[x] for x in topic_words_ids]) + '\n')

    args.save(join(model_dir, 'config.yml'))


class TopicInferer:

    def __init__(self, model_dir, verbose=True):
        self._model_dir = model_dir
        self._verbose = verbose
        if self._verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self._params = LDAArgs.load(join(model_dir, 'config.yml'))
        self._ldamodel = gensim.models.LdaMulticore.load(join(model_dir, 'LDA.model'))

    def _init_words_per_topics(self, words_per_topic):
        topic_word_dict = {}
        for t_id in range(self._params.num_topics):
            topic_words_ids = [x[0] for x in
                               self._ldamodel.get_topic_terms(
                                   t_id,
                                   topn=(words_per_topic or self._params.words_per_topic))]
            topic_word_dict[t_id] = [self._ldamodel.id2word[x] for x in topic_words_ids]

        return topic_word_dict

    def from_collection(self, test_collection, dialogue_as_doc=False, words_per_topic=None):
        topic_word_dict = self._init_words_per_topics(words_per_topic)
        output = []
        for lno, sample in enumerate(test_collection):
            utterances = sample.strip().split('\t')

            if dialogue_as_doc:
                words = ' '.join(utterances).split()
                dialogue_terms = analyzer.normalize_sequence(words)

                doc = self._ldamodel.id2word.doc2bow(dialogue_terms)
                topic_ids = self._ldamodel.get_document_topics(doc)
                if len(topic_ids) > 0:
                    t_id = sorted(topic_ids, key=lambda x: x[1], reverse=True)[0][0]
                    output.append((t_id, topic_word_dict[t_id]))
                else:
                    output.append((-1, []))
            else:
                t_ids, t_words = [], []
                for i, utterance in enumerate(utterances):
                    utterance_terms = analyzer.normalize_sequence(utterance.split())

                    doc = self._ldamodel.id2word.doc2bow(utterance_terms)
                    topic_ids = self._ldamodel.get_document_topics(doc)
                    if len(topic_ids) > 0:
                        t_id = sorted(topic_ids, key=lambda x: x[1], reverse=True)[0][0]
                        t_ids.append(t_id)
                        t_words.append(topic_word_dict[t_id])
                    else:
                        t_ids.append(-1)
                        t_words.append([])

                output.append((t_ids, t_words))

        return output

    def from_file(self, test_data, output_file, dialogue_as_doc=False, words_per_topic=None):
        topic_word_dict = self._init_words_per_topics(words_per_topic)

        if output_file is None:
            output_file = fs.replace_ext(test_data, 'topical.txt')

        sw = Stopwatch()

        with codecs.getreader('utf-8')(open(test_data, 'rb')) as test_file:
            with codecs.getwriter('utf-8')(open(output_file, 'wb')) as out_file:
                for lno, line in enumerate(test_file):
                    utterances = line.strip().split('\t')
                    out_file.write(line.strip() + "\t")

                    if lno % 100000 == 0:
                        sw.print('  {} lines inferred'.format(lno))

                    if dialogue_as_doc:
                        words = ' '.join(utterances[:-1]).split()
                        dialogue_terms = analyzer.normalize_sequence(words)

                        doc = self._ldamodel.id2word.doc2bow(dialogue_terms)
                        topic_ids = self._ldamodel.get_document_topics(doc)
                        if len(topic_ids) > 0:
                            t_id = sorted(topic_ids, key=lambda x: x[1], reverse=True)[0][0]
                            out_file.write(' '.join(topic_word_dict[t_id]))
                        else:
                            out_file.write('<NO_TOPIC>')
                    else:
                        for i, utterance in enumerate(utterances):
                            utterance_terms = analyzer.normalize_sequence(utterance.split())

                            doc = self._ldamodel.id2word.doc2bow(utterance_terms)
                            topic_ids = self._ldamodel.get_document_topics(doc)
                            if len(topic_ids) > 0:
                                t_id = sorted(topic_ids, key=lambda x: x[1], reverse=True)[0][0]
                                out_file.write(
                                    ' '.join(topic_word_dict[t_id]) + ('\t' if i < len(utterances) - 1 else ''))
                            else:
                                out_file.write('<NO_TOPIC>')

                    out_file.write('\n')

        sw.print('Done!!!')


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=("train", "infer"), help='mode')
    parser.add_argument('--model_dir', type=str, required=True, help='model directory')
    parser.add_argument('--data', type=str,
                        help='data (if directory, each document is a file, or else each document is a line)')
    parser.add_argument('--n_topics', type=int, default=200, help='number of topics')
    parser.add_argument('--no_below', type=int, default=600,
                        help='terms with frequency lower than this argument would be dropped')
    parser.add_argument('--min_length', type=int, help='min length of words')
    parser.add_argument('--test_data', type=str, help='test data')
    parser.add_argument('--dialogue_as_doc', action='store_true', help='treats whole dialogue as document')
    parser.add_argument('--output', type=str, help='output file')

    args = parser.parse_args()
    if args.mode == 'train':
        _params = {
            "num_topics": args.n_topics,
            "documents": args.data,
            "no_below": args.no_below,
            "min_length": args.min_length,
            "passes": 70,
            "eval_every": 10,
            "chunksize": 10000,
            "iterations": 1000,
            "words_per_topic": 100
        }

        print("Training starts with arguments: {}".format(_params))
        train(args.model_dir, LDAArgs(_params))
    elif args.mode == 'infer':
        TopicInferer(args.model_dir).from_file(args.test_data, args.output, args.dialogue_as_doc)


if __name__ == "__main__":
    main()
