import tensorflow as tf

from ..model_helper import TrainModel, EvalModel, InferModel
from .thred_iterators import get_iterator, get_infer_iterator
from .thred_model import TopicAwareHierarchicalSeq2SeqModel
from thred.util import vocab


def create_train_model(hparams, scope=None, num_workers=1, jobid=0):
    """Create train graph, model, and iterator."""

    graph = tf.Graph()

    vocab.create_vocabulary(hparams.vocab_file, hparams.train_data, hparams.vocab_size)

    with graph.as_default(), tf.container(scope or "train"):
        vocab_table = vocab.create_vocab_table(hparams.vocab_file)

        dataset = tf.data.TextLineDataset(hparams.train_data)
        skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

        iterator = get_iterator(
            dataset,
            vocab_table,
            hparams.batch_size,
            hparams.num_turns,
            hparams.num_buckets,
            hparams.topic_words_per_utterance,
            hparams.src_max_len,
            hparams.tgt_max_len,
            skip_count=skip_count_placeholder,
            num_shards=num_workers,
            shard_index=jobid)

        # Note: One can set model_device_fn to
        # `tf.train.replica_device_setter(ps_tasks)` for distributed training.
        model_device_fn = None
        # if extra_args: model_device_fn = extra_args.model_device_fn
        with tf.device(model_device_fn):
            model = TopicAwareHierarchicalSeq2SeqModel(
                mode=tf.contrib.learn.ModeKeys.TRAIN,
                iterator=iterator,
                num_turns=hparams.num_turns,
                params=hparams,
                scope=scope)

    return TrainModel(graph=graph,
                      model=model,
                      iterator=iterator,
                      skip_count_placeholder=skip_count_placeholder)


def create_pretrain_model(hparams, scope=None, num_workers=1, jobid=0):
    """Create train graph, model, and iterator."""
    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "pretrain"):
        vocab_table = vocab.create_vocab_table(hparams.vocab_file)

        iterator = get_iterator(
            hparams.pretrain_data,
            vocab_table,
            hparams.batch_size,
            hparams.num_pretrain_turns,
            hparams.num_buckets,
            hparams.topic_words_per_utterance,
            hparams.src_max_len,
            hparams.tgt_max_len,
            num_shards=num_workers,
            shard_index=jobid)

        model = TopicAwareHierarchicalSeq2SeqModel(
            mode=tf.contrib.learn.ModeKeys.TRAIN,
            iterator=iterator,
            num_turns=hparams.num_pretrain_turns,
            params=hparams,
            scope=scope,
            log_trainables=False)

    return TrainModel(
        graph=graph,
        model=model,
        iterator=iterator,
        skip_count_placeholder=None)


def create_eval_model(hparams, scope=None):
    """Create train graph, model, src/tgt file holders, and iterator."""

    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "eval"):
        vocab_table = vocab.create_vocab_table(hparams.vocab_file)
        eval_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)

        eval_dataset = tf.data.TextLineDataset(eval_file_placeholder)
        iterator = get_iterator(
            eval_dataset,
            vocab_table,
            hparams.batch_size,
            hparams.num_turns,
            hparams.num_buckets,
            hparams.topic_words_per_utterance,
            hparams.src_max_len,
            hparams.tgt_max_len)

        model = TopicAwareHierarchicalSeq2SeqModel(
            mode=tf.contrib.learn.ModeKeys.EVAL,
            iterator=iterator,
            num_turns=hparams.num_turns,
            params=hparams,
            scope=scope,
            log_trainables=False)

    return EvalModel(
        graph=graph,
        model=model,
        eval_file_placeholder=eval_file_placeholder,
        iterator=iterator)


def create_infer_model(hparams, scope=None):
    """Create inference model."""
    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "infer"):
        vocab_table = vocab.create_vocab_table(hparams.vocab_file)
        reverse_vocab_table = vocab.create_rev_vocab_table(hparams.vocab_file)

        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

        src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)

        iterator = get_infer_iterator(
            src_dataset,
            vocab_table,
            batch_size=batch_size_placeholder,
            num_turns=hparams.num_turns,
            topic_words_per_utterance=hparams.topic_words_per_utterance,
            src_max_len=hparams.src_max_len)

        model = TopicAwareHierarchicalSeq2SeqModel(
            mode=tf.contrib.learn.ModeKeys.INFER,
            iterator=iterator,
            num_turns=hparams.num_turns,
            params=hparams,
            rev_vocab_table=reverse_vocab_table,
            scope=scope,
            log_trainables=False)

    return InferModel(
        graph=graph,
        model=model,
        src_placeholder=src_placeholder,
        batch_size_placeholder=batch_size_placeholder,
        iterator=iterator)
