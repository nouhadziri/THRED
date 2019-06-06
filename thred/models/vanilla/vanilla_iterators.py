import tensorflow as tf

from ..model_helper import BatchedInput
from thred.util import vocab


def get_iterator(dataset,
                 vocab_table,
                 batch_size,
                 num_buckets,
                 random_seed=None,
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
        utterances = tf.string_split([line], delimiter="\t").values
        i, sp = tf.constant(0), tf.Variable([], dtype=tf.string)
        cond = lambda i, sp: tf.less(i, tf.size(utterances)-1)

        def loop_body(i, sp):
            splitted = tf.string_split([utterances[i]]).values
            if src_max_len:
                splitted = tf.cond(tf.less(i, tf.size(utterances)-2),
                                   lambda: splitted[:src_max_len - 1],
                                   lambda: splitted[:src_max_len])

            splitted = tf.cond(tf.less(i, tf.size(utterances)-2),
                               lambda: tf.concat([splitted, [vocab.SEP]], axis=0),
                               lambda: splitted)

            return tf.add(i, 1), tf.concat([sp, splitted], axis=0)

        _, srcs = tf.while_loop(cond, loop_body, [i, sp], shape_invariants=[i.get_shape(), tf.TensorShape([None])])


        # srcs = [tf.string_split([utterances[t]]).values for t in range(num_inputs)]
        tgt = tf.string_split([utterances[tf.size(utterances)-1]]).values
        aggregated_src = tf.reduce_join([srcs], axis=0, separator=" ")

        return aggregated_src, tgt[:tgt_max_len] if tgt_max_len else tgt

    src_tgt_dataset = src_tgt_dataset.map(tokenize,
                                          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(vocab_table.lookup(src), tf.int32),
                          tf.cast(vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src,
                          tf.concat(([sos_id], tgt), 0),
                          tf.concat((tgt, [eos_id]), 0)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    # Add in sequence lengths.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
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
                tf.TensorShape([]),  # src_len
                tf.TensorShape([])),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                eos_id,  # src
                eos_id,  # tgt_input
                eos_id,  # tgt_output
                0,  # src_len -- unused
                0))  # tgt_len -- unused

    if num_buckets > 1:

        def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
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
    (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
     tgt_seq_len) = (batched_iter.get_next())
    return BatchedInput(
        initializer=batched_iter.initializer,
        sources=src_ids,
        target_input=tgt_input_ids,
        target_output=tgt_output_ids,
        source_sequence_lengths=src_seq_len,
        target_sequence_length=tgt_seq_len)


def get_infer_iterator(test_dataset,
                       vocab_table,
                       batch_size,
                       src_max_len=None):
    eos_id = tf.constant(vocab.EOS_ID, dtype=tf.int32)

    def tokenize(line):
        utterances = tf.string_split([line], delimiter="\t").values
        i, sp = tf.constant(0), tf.Variable([], dtype=tf.string)
        cond = lambda i, sp: tf.less(i, tf.size(utterances) - 1)

        def loop_body(i, sp):
            splitted = tf.string_split([utterances[i]]).values
            if src_max_len:
                splitted = tf.cond(tf.less(i, tf.size(utterances) - 2),
                                   lambda: splitted[:src_max_len - 1],
                                   lambda: splitted[:src_max_len])

            splitted = tf.cond(tf.less(i, tf.size(utterances) - 2),
                               lambda: tf.concat([splitted, [vocab.SEP]], axis=0),
                               lambda: splitted)

            return tf.add(i, 1), tf.concat([sp, splitted], axis=0)

        _, srcs = tf.while_loop(cond, loop_body, [i, sp], shape_invariants=[i.get_shape(), tf.TensorShape([None])])

        aggregated_src = tf.reduce_join([srcs], axis=0, separator=" ")
        return aggregated_src

    test_dataset = test_dataset.map(tokenize)

    if src_max_len:
        test_dataset = test_dataset.map(lambda src: src[:src_max_len])
    # Convert the word strings to ids
    test_dataset = test_dataset.map(lambda src: tf.cast(vocab_table.lookup(src), tf.int32))

    # Add in the word counts.
    test_dataset = test_dataset.map(lambda src: (src, tf.size(src)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The entry is the source line rows;
            # this has unknown-length vectors.  The last entry is
            # the source row size; this is a scalar.
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([])),  # src_len
            # Pad the source sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                eos_id,  # src
                0))  # src_len -- unused

    batched_dataset = batching_func(test_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, src_seq_len) = batched_iter.get_next()
    return BatchedInput(
        initializer=batched_iter.initializer,
        sources=src_ids,
        target_input=None,
        target_output=None,
        source_sequence_lengths=src_seq_len,
        target_sequence_length=None)
