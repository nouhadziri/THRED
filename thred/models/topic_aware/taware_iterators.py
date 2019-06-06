import tensorflow as tf

from thred.models.model_helper import TopicalBatchedInput as BatchedInput
from thred.util import vocab


def get_iterator(dataset,
                 vocab_table,
                 batch_size,
                 num_buckets,
                 random_seed=None,
                 topic_words_per_utterance=None,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0):
    if not output_buffer_size:
        output_buffer_size = batch_size * 1000

    eos_id = tf.constant(vocab.EOS_ID, dtype=tf.int32)
    sos_id = tf.constant(vocab.SOS_ID, dtype=tf.int32)

    src_tgt_dataset = dataset.shard(num_shards, shard_index)
    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, random_seed)

    def tokenize(line):
        delimited_line = tf.string_split([line], delimiter="\t").values
        # utterances = tf.string_split([tf.py_func(lambda x: x.strip(), [delimited_line[0]], [tf.string])[0]],
        #                              delimiter="\t").values
        # topics = tf.string_split([tf.py_func(lambda x: x.strip(), [delimited_line[1]], [tf.string])[0]],
        #                          delimiter="\t").values
        topics = tf.string_split([delimited_line[-1]]).values

        i, sp = tf.constant(0), tf.Variable([], dtype=tf.string)
        cond = lambda i, sp: tf.less(i, tf.size(delimited_line) - 2)

        def loop_body(i, sp):
            splitted = tf.string_split([delimited_line[i]]).values
            if src_max_len:
                splitted = tf.cond(tf.less(i, tf.size(delimited_line) - 3),
                                   lambda: splitted[:src_max_len - 1],
                                   lambda: splitted[:src_max_len])

            splitted = tf.cond(tf.less(i, tf.size(delimited_line) - 3),
                               lambda: tf.concat([splitted, [vocab.SEP]], axis=0),
                               lambda: splitted)

            return tf.add(i, 1), tf.concat([sp, splitted], axis=0)

        _, srcs = tf.while_loop(cond, loop_body, [i, sp], shape_invariants=[i.get_shape(), tf.TensorShape([None])])

        # srcs = [tf.string_split([utterances[t]]).values for t in range(num_inputs)]
        tgt = tf.string_split([delimited_line[tf.size(delimited_line) - 2]]).values
        aggregated_src = tf.reduce_join([srcs], axis=0, separator=" ")

        return aggregated_src, \
               tgt[:tgt_max_len] if tgt_max_len else tgt, \
               tf.string_split([topics[0]]).values

    src_tgt_dataset = src_tgt_dataset.map(tokenize,
                                          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, tgt, topic: tf.logical_and(tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0), tf.size(topic) > 0))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt, topic: (src[:src_max_len], tgt, topic),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt, topic: (src, tgt[:tgt_max_len], topic),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if topic_words_per_utterance:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt, topic: (src, tgt, topic[:topic_words_per_utterance]),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt, topic: (tf.cast(vocab_table.lookup(src), tf.int32),
                                 tf.cast(vocab_table.lookup(tgt), tf.int32),
                                 tf.cast(vocab_table.lookup(topic), tf.int32)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt, topic: (src,
                                 tf.concat(([sos_id], tgt), 0),
                                 tf.concat((tgt, [eos_id]), 0),
                                 topic),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    # Add in sequence lengths.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out, topic: (
            src, tgt_in, tgt_out, topic, tf.size(src), tf.size(tgt_in), tf.size(topic)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # tgt_input
                tf.TensorShape([None]),  # tgt_output
                tf.TensorShape([None]),  # topic
                tf.TensorShape([]),  # src_len
                tf.TensorShape([]),  # tgt_len
                tf.TensorShape([])),  # topic_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                eos_id,  # src
                eos_id,  # tgt_input
                eos_id,  # tgt_output
                eos_id,  # topic
                0,  # src_len -- unused
                0,  # tgt_len -- unused
                0))  # topic_len -- unused

    if num_buckets > 1:

        def key_func(src_unused, tgt_in_unused, tgt_out_unused, topic_unused, src_len, tgt_len, topic_len_unused):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            if src_max_len:
                bucket_width = (src_max_len + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = src_tgt_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    else:
        batched_dataset = batching_func(src_tgt_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_input_ids, tgt_output_ids, topic_ids,
     src_seq_len, tgt_seq_len, topic_seq_len) = (batched_iter.get_next())
    return BatchedInput(
        initializer=batched_iter.initializer,
        sources=src_ids,
        target_input=tgt_input_ids,
        target_output=tgt_output_ids,
        topic=topic_ids,
        source_sequence_lengths=src_seq_len,
        target_sequence_length=tgt_seq_len,
        topic_sequence_length=topic_seq_len)


def get_infer_iterator(test_dataset,
                       vocab_table,
                       batch_size,
                       topic_words_per_utterance=None,
                       src_max_len=None):
    eos_id = tf.constant(vocab.EOS_ID, dtype=tf.int32)

    def tokenize(line):
        delimited_line = tf.string_split([line], delimiter="\t").values
        # utterances = tf.string_split([tf.py_func(lambda x: x.strip(), [delimited_line[0]], [tf.string])[0]],
        #                              delimiter="\t").values
        # topics = tf.string_split([tf.py_func(lambda x: x.strip(), [delimited_line[1]], [tf.string])[0]],
        #                          delimiter="\t").values
        topics = tf.string_split([delimited_line[-1]]).values

        i, sp = tf.constant(0), tf.Variable([], dtype=tf.string)
        cond = lambda i, sp: tf.less(i, tf.size(delimited_line) - 2)

        def loop_body(i, sp):
            splitted = tf.string_split([delimited_line[i]]).values

            if src_max_len:
                splitted = tf.cond(tf.less(i, tf.size(delimited_line) - 3),
                                   lambda: splitted[:src_max_len - 1],
                                   lambda: splitted[:src_max_len])

            splitted = tf.cond(tf.less(i, tf.size(delimited_line) - 3),
                               lambda: tf.concat([splitted, [vocab.SEP]], axis=0),
                               lambda: splitted)

            return tf.add(i, 1), tf.concat([sp, splitted], axis=0)

        _, srcs = tf.while_loop(cond, loop_body, [i, sp], shape_invariants=[i.get_shape(), tf.TensorShape([None])])
        aggregated_src = tf.reduce_join([srcs], axis=0, separator=" ")

        return aggregated_src, tf.string_split([topics[0]]).values

    test_dataset = test_dataset.map(tokenize)

    if src_max_len:
        test_dataset = test_dataset.map(lambda src, topic: (src[:src_max_len], topic))

    if topic_words_per_utterance:
        test_dataset = test_dataset.map(
            lambda src, topic: (src, topic[:topic_words_per_utterance]))
    # Convert the word strings to ids
    test_dataset = test_dataset.map(lambda src, topic: (tf.cast(vocab_table.lookup(src), tf.int32),
                                                        tf.cast(vocab_table.lookup(topic), tf.int32)))

    # Add in the word counts.
    test_dataset = test_dataset.map(lambda src, topic: (src, topic, tf.size(src), tf.size(topic)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The entry is the source line rows;
            # this has unknown-length vectors.  The last entry is
            # the source row size; this is a scalar.
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # topic
                tf.TensorShape([]),  # src_len
                tf.TensorShape([])),  # topic_len
            # Pad the source sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                eos_id,  # src
                eos_id,  # topic
                0,  # src_len -- unused
                0))  # topic_len -- unused

    batched_dataset = batching_func(test_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, topic_ids, src_seq_len, topic_seq_len) = batched_iter.get_next()
    return BatchedInput(
        initializer=batched_iter.initializer,
        sources=src_ids,
        target_input=None,
        target_output=None,
        topic=topic_ids,
        source_sequence_lengths=src_seq_len,
        target_sequence_length=None,
        topic_sequence_length=topic_seq_len)
