import codecs
import collections
import logging
import os
import random
from collections import defaultdict

from thred.util import fs
from thred.util.kv import TinyRedis
from thred.util.misc import Stopwatch
from thred.util.summary_statistics import SampledSummaryStat

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger('corpus_toolkit')

SEPARATOR = "\t"


class DialogueCorpus:

    def __init__(self, data_path, utterance_sep=SEPARATOR):
        super(DialogueCorpus, self).__init__()
        self.data_path = data_path
        self.utterance_sep = utterance_sep

    def iterate_over(self, utterance_consumer):
        lno = 0

        with codecs.getreader("utf-8")(open(self.data_path, mode="rb")) as data_file:
            for line in data_file:
                lno += 1
                for i, utter in enumerate(line.split(self.utterance_sep)):
                    utterance_consumer(lno, i, utter)

        return lno


def __build_vocabulary(dialogue_corpus, steps_per_log=100000):
    vocabulary = set()

    def consume(lno, _, utterance):
        if lno % steps_per_log == 0:
            logger.info('{} lines processed - so far vocab {}'.format(lno, len(vocabulary)))

        for w in utterance.split():
            vocabulary.add(w)

    line_number = dialogue_corpus.iterate_over(consume)
    return vocabulary, line_number


def preprocess_for_lda(dialogue_corpus, output_path,
                       n_frequents_to_drop=500, min_utterance_length=3, min_word_length=3,
                       ngrams_path=None, steps_per_log=100000):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    elif not os.path.isdir(output_path):
        raise ValueError('output must be a directory: ' + output_path)

    tf_dict = defaultdict(int)

    sw = Stopwatch()

    ngrams_dict = {}
    if ngrams_path:
        with codecs.getreader('utf-8')(open(ngrams_path, 'rb')) as ngrams_file:
            for line in ngrams_file:
                ngram, count = tuple(line.strip().split('\t'))
                ngrams_dict[ngram] = (int(count), [])

        logger.info('{} ngrams provided to drop'.format(len(ngrams_dict)))

    n_utterances = 0

    logger.info('[Pass 1] finding frequent words...')
    with codecs.getreader("utf-8")(open(dialogue_corpus.data_path, mode="rb")) as data_file:
        lno = 0
        for line in data_file:
            lno += 1
            line = line.strip()

            if lno % steps_per_log == 0:
                logger.info('{} lines processed - so far vocab {} - time {}'.format(lno, len(tf_dict), sw.elapsed()))

            utterances = line.split(dialogue_corpus.utterance_sep)
            n_utterances = max(n_utterances, len(utterances))
            for utter in utterances:
                tokens = utter.split()
                for w in tokens:
                    if len(w) >= min_word_length:
                        tf_dict[w] += 1

    sorted_vocab = sorted(tf_dict, key=tf_dict.get, reverse=True)
    sorted_vocab = set(sorted_vocab[n_frequents_to_drop:])

    lines_to_drop = set()
    if ngrams_dict:
        logger.info('[Pass 2] finding dialogues containing ngrams to drop...')

        processed_lines, lno = 0, 0
        with codecs.getreader("utf-8")(open(dialogue_corpus.data_path, mode="rb")) as data_file:
            for line in data_file:
                lno += 1
                line = line.strip()

                if lno % steps_per_log == 0:
                    logger.info(
                        '{} lines processed - {} will be chosen - time {}'.format(
                            lno, processed_lines, sw.elapsed()))

                empty_found = False
                nontrivial_len = 0

                filtered_words = [w.strip() for w in line.split() if w in sorted_vocab]
                if len(filtered_words) >= min_utterance_length:
                    nontrivial_len += len(filtered_words)
                else:
                    empty_found = True

                if not empty_found:
                    processed_lines += 1
                    for ngram, container in ngrams_dict.items():
                        if ngram in line:
                            container[1].append((line, nontrivial_len))

        for ngram, container in ngrams_dict.items():
            # sorted_ngram_lines = sorted(container[1], key=lambda x: x[1])
            lines = set(line for line, _ in container[1])
            already_to_drop = lines_to_drop.intersection(lines)

            if already_to_drop:
                selectable_lines = lines.difference(already_to_drop)
                n_to_drop = max(container[0] - len(already_to_drop), 0)
            else:
                selectable_lines = lines
                n_to_drop = container[0]

            dropped_lines = random.sample(selectable_lines, min(n_to_drop, len(selectable_lines)))
            for line in dropped_lines:
                lines_to_drop.add(line)

        logger.info('[Pass 2] done with {} lines chosen for tossing out'.format(len(lines_to_drop)))

    logger.info('[Final Pass] generating processed data...')

    out_file = fs.replace_dir(dialogue_corpus.data_path, output_path, 'lda.t')
    writer = codecs.getwriter("utf-8")(open(out_file, mode="wb"))

    processed_lines = 0
    with codecs.getwriter("utf-8")(open(fs.replace_dir(dialogue_corpus.data_path, output_path, 'processed'), mode="wb")) \
            as processed_file:
        lno = 0
        with codecs.getreader("utf-8")(open(dialogue_corpus.data_path, mode="rb")) as data_file:
            processed_data, lda_data = [], []
            for line in data_file:
                lno += 1
                line = line.strip()

                if lno % steps_per_log == 0:
                    for dialog, prc_line in zip(lda_data, processed_data):
                        writer.write('\t'.join(dialog) + '\n')
                        processed_file.write(prc_line)
                    logger.info(
                        '{} lines processed - {} flushed - {} chosen - time {}'.format(
                            lno, len(processed_data), processed_lines, sw.elapsed()))
                    processed_data, lda_data = [], []

                empty_found = False
                processed_dialog = []

                if line in lines_to_drop:
                    continue

                filtered_words = [w.strip() for w in line.split() if w in sorted_vocab]
                if len(filtered_words) >= min_utterance_length:
                    processed_dialog = [' '.join(filtered_words)]
                else:
                    empty_found = True

                if not empty_found:
                    processed_lines += 1
                    lda_data.append(processed_dialog)
                    processed_data.append(line + '\n')

            for dialog, prc_line in zip(lda_data, processed_data):
                writer.write('\t'.join(dialog) + '\n')
                processed_file.write(prc_line)

        writer.close()
        logger.info('{} of {} processed, finished in {}'.format(processed_lines, lno, sw.elapsed()))


class AnalysisArgs(
    collections.namedtuple("AnalysisArgs",
                           ("n_frequent_words", "n_rare_words",
                            "min_freq", "vocab_size",
                            "save_tf"))):
    pass


def rank_ngrams(dialogue_corpus, ngrams, redis_port, steps_per_log=100000):
    if not ngrams:
        raise ValueError('ngrams is required')

    ngrams_cache = {}
    redis = TinyRedis(port=redis_port, max_connections=1000)

    for ngram in ngrams:
        ngrams_cache[ngram] = (redis, defaultdict(int))

    dir, fname, _ = fs.split3(dialogue_corpus.data_path)

    sw = Stopwatch()
    with codecs.getreader("utf-8")(open(dialogue_corpus.data_path, mode="rb")) as data_file:
        lno = 0
        for line in data_file:
            lno += 1

            if lno % steps_per_log == 0:
                if ngrams_cache:
                    for ngram in ngrams_cache:
                        logger.info('{}-grams flushing {} keys...'.format(ngram, len(ngrams_cache[ngram][1])))
                        ngrams_cache[ngram][0].pl_hincrby('{}#{}'.format(ngram, fname), ngrams_cache[ngram][1])
                        ngrams_cache[ngram] = (ngrams_cache[ngram][0], defaultdict(int))
                    logger.info('ngrams flushed...')

                logger.info('{} lines processed - time {}'.format(lno, sw.elapsed()))

            for t, utter in enumerate(line.split(dialogue_corpus.utterance_sep)):
                tokens = utter.split()
                for ngram in ngrams_cache:
                    for i in range(len(tokens) - ngram):
                        ngrams_cache[ngram][1][' '.join(tokens[i:i + ngram])] += 1

    for ngram in ngrams_cache:
        logger.info('{}-grams flushing {} keys...'.format(ngram, len(ngrams_cache[ngram][1])))
        ngrams_cache[ngram][0].pl_hincrby('{}#{}'.format(ngram, fname), ngrams_cache[ngram][1])
        ngrams_cache[ngram] = (ngrams_cache[ngram][0], defaultdict(int))
    logger.info('ngrams flushed...')

    for ngram in ngrams_cache:
        with codecs.getwriter("utf-8")(open(os.path.join(dir, '{}.{}grams'.format(fname, ngram)), 'wb')) as ngram_file:
            for ngram_str, freq in ngrams_cache[ngram][0].hscan('{}#{}'.format(ngram, fname)):
                if int(freq) > 10:
                    ngram_file.write('{}\t{}\n'.format(ngram_str, freq))

    next(iter(ngrams_cache.values()))[0].close()


def count_ngrams(dialogue_corpus, ngrams, redis_port, steps_per_log=5000):
    if not ngrams:
        raise ValueError('ngrams is required')

    _, fname, _ = fs.split3(dialogue_corpus.data_path)

    def get_key(n):
        return '{}__c{}'.format(fname, n)

    ngrams_cache = {}
    n_words = 0
    with TinyRedis(port=redis_port, max_connections=100) as redis:

        for ngram in ngrams:
            redis.delete(get_key(ngram))
            ngrams_cache[ngram] = set()

        sw = Stopwatch()
        with codecs.getreader("utf-8")(open(dialogue_corpus.data_path, mode="rb")) as data_file:
            lno = 0
            for line in data_file:
                lno += 1

                if lno % steps_per_log == 0:
                    if ngrams_cache:
                        for ngram in ngrams_cache:
                            logger.info('{}-grams flushing {} keys...'.format(ngram, len(ngrams_cache[ngram])))
                            redis.pfadd(get_key(ngram), *ngrams_cache[ngram])
                            ngrams_cache[ngram] = set()
                        logger.info('ngrams flushed...')

                    logger.info('{} lines processed - time {}'.format(lno, sw.elapsed()))

                for utter in line.split(dialogue_corpus.utterance_sep):
                    tokens = utter.split()
                    n_words += len(tokens)
                    for ngram in ngrams_cache:
                        for i in range(len(tokens) - ngram):
                            ngrams_cache[ngram].add(' '.join(tokens[i:i + ngram]))

        for ngram in ngrams_cache:
            logger.info('{}-grams flushing {} keys...'.format(ngram, len(ngrams_cache[ngram])))
            redis.pfadd(get_key(ngram), *ngrams_cache[ngram])
        logger.info('ngrams flushed...')

        logger.info('**** {} ****'.format(fname))
        logger.info("#words = {}".format(n_words))
        for ngram in ngrams:
            ngram_cnt = redis.pfcount(get_key(ngram))
            logger.info('# {}-grams = {} | distinct-{} = {:.3f}'.format(ngram, ngram_cnt, ngram, ngram_cnt / n_words))


def analyze(dialogue_corpus, analysis_args, steps_per_log=100000):
    dir, fname, _ = fs.split3(dialogue_corpus.data_path)

    if analysis_args.n_frequent_words > 0:
        frequent_words_path = os.path.join(dir, '{}.top{}'.format(fname, analysis_args.n_frequent_words))
        frequent_words_file = codecs.getwriter("utf-8")(open(frequent_words_path, mode="wb"))
    else:
        frequent_words_file = None

    if analysis_args.n_rare_words > 0:
        rare_words_path = os.path.join(dir, '{}.bottom{}'.format(fname, analysis_args.n_rare_words))
        rare_words_file = codecs.getwriter("utf-8")(open(rare_words_path, mode="wb"))
    else:
        rare_words_file = None

    if analysis_args.save_tf:
        tf_path = os.path.join(dir, '{}.tf'.format(fname))
        tf_file = codecs.getwriter("utf-8")(open(tf_path, mode="wb"))
    else:
        tf_file = None

    tf_dict = defaultdict(int)

    lno, uno = 0, 0
    wno_stat_per_turn = []
    wno_stat = SampledSummaryStat()
    sw = Stopwatch()

    with codecs.getreader("utf-8")(open(dialogue_corpus.data_path, mode="rb")) as data_file:
        for line in data_file:
            lno += 1

            if lno % steps_per_log == 0:
                logger.info('{} lines processed - so far vocab {} - time {}'.format(lno, len(tf_dict), sw.elapsed()))

            for t, utter in enumerate(line.split(dialogue_corpus.utterance_sep)):
                uno += 1
                tokens = utter.split()
                for w in tokens:
                    tf_dict[w] += 1

                if t >= len(wno_stat_per_turn):
                    wno_stat_per_turn.append(SampledSummaryStat())
                wno_stat_per_turn[t].accept(len(tokens))
                wno_stat.accept(len(tokens))

    sorted_vocab = sorted(tf_dict, key=tf_dict.get, reverse=True)

    min_freq_vocab_size, min_freq_vol_size = 0, 0
    vol_size = 0
    for i, w in enumerate(sorted_vocab):
        tf = tf_dict[w]

        if tf >= analysis_args.min_freq:
            min_freq_vocab_size += 1
            min_freq_vol_size += tf

        if i < analysis_args.vocab_size:
            vol_size += tf

        if tf_file:
            tf_file.write('{}\t{}\n'.format(w, tf))
        if frequent_words_file and i < analysis_args.n_frequent_words:
            frequent_words_file.write('{}\n'.format(w))
        if rare_words_file and i > len(sorted_vocab) - analysis_args.n_rare_words:
            rare_words_file.write('{}\n'.format(w))

    if frequent_words_file:
        frequent_words_file.close()

    if rare_words_file:
        rare_words_file.close()

    if tf_file:
        tf_file.close()

    print('**** {} ****'.format(os.path.abspath(dialogue_corpus.data_path)))
    print('lines {} | utterances {} | vocab {} tf {}'.format(lno, uno, len(tf_dict), wno_stat.get_sum()))
    print('min_freq {} -> {}/{} {:.1f}% - {}/{} {:.1f}%)'.format(
        analysis_args.min_freq,
        min_freq_vocab_size, len(tf_dict),
        100.0 * min_freq_vocab_size / len(tf_dict),
        min_freq_vol_size, wno_stat.get_sum(),
        100.0 * min_freq_vol_size / wno_stat.get_sum()))
    print('vocab_size {}/{} {:.1f}% - {}/{} {:.1f}%)'.format(
        analysis_args.vocab_size, len(tf_dict),
        100.0 * analysis_args.vocab_size / len(tf_dict),
        vol_size, wno_stat.get_sum(),
        100.0 * vol_size / wno_stat.get_sum()))

    print('utterances per line: {:.1f}'.format(uno / lno))
    print('utterance_len: avg {:.1f} - stdev {:.1f} - median {} - min {} - max {}'.format(
        wno_stat.get_average(),
        wno_stat.get_stdev(),
        wno_stat.get_median(),
        wno_stat.get_min(),
        wno_stat.get_max()))
    print('utterance_len per turn')
    for t, stat in enumerate(wno_stat_per_turn):
        print('  turn {} - avg {:.1f} - stdev {:.1f} - median {} - min {} - max {}'.format(
            t,
            stat.get_average(),
            stat.get_stdev(),
            stat.get_median(),
            stat.get_min(),
            stat.get_max()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="types of operation")
    s_group = subparsers.add_parser("sync")
    r_group = subparsers.add_parser("analyze")
    n_group = subparsers.add_parser("ngrams")
    p_group = subparsers.add_parser("preprocess-lda")

    parser.add_argument('-d', '--data', type=str, required=True,
                        help="data path")
    parser.add_argument('-s', '--separator', type=str, default=SEPARATOR,
                        help="utterance separator")

    r_group.add_argument('--n_frequents', default=-1, type=int)
    r_group.add_argument('--n_rares', default=-1, type=int)
    r_group.add_argument('--vocab_size', default=0, type=int)
    r_group.add_argument('--min_freq', default=1, type=int)
    r_group.add_argument('--save_tf', action='store_true')
    r_group.set_defaults(op=lambda: "analyze")

    n_group.add_argument('-n', '--ngrams', nargs='+', type=int)
    n_group.add_argument('--ngram_redis_port', default=6389, type=int)
    n_group.add_argument('--operation', default='count', choices=("count", "rank"), type=str)
    n_group.set_defaults(op=lambda: "ngrams")

    p_group.add_argument('--output', required=True, type=str)
    p_group.add_argument('--min_word_length', default=3, type=int)
    p_group.add_argument('--min_utterance_length', default=3, type=int)
    p_group.add_argument('--n_frequents_to_drop', default=400, type=int)
    p_group.add_argument('--ngrams_file', type=str)
    p_group.set_defaults(op=lambda: "preprocess-lda")

    args = parser.parse_args()
    corpus = DialogueCorpus(args.data, args.separator)

    if args.op() == "analyze":
        analyze(corpus, AnalysisArgs(args.n_frequents, args.n_rares, args.min_freq, args.vocab_size, args.save_tf))
    elif args.op() == "ngrams":
        if args.operation == "rank":
            rank_ngrams(corpus, args.ngrams, args.ngram_redis_port)
        else:
            count_ngrams(corpus, args.ngrams, args.ngram_redis_port)
    elif args.op() == "preprocess-lda":
        preprocess_for_lda(corpus, args.output, args.n_frequents_to_drop,
                           args.min_utterance_length, args.min_word_length, args.ngrams_file)
    else:
        raise ValueError('Unknown operation')
