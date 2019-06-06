import tensorflow as tf

from thred.models.model_helper import BatchedInput
from thred.util import vocab


def get_iterator(dataset,
                 vocab_table,
                 batch_size,
                 num_turns,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 random_seed=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0):
    num_inputs = num_turns - 1

    if not output_buffer_size:
        output_buffer_size = batch_size * 1000

    eos_id = tf.constant(vocab.EOS_ID, dtype=tf.int32)
    sos_id = tf.constant(vocab.SOS_ID, dtype=tf.int32)

    src_tgt_dataset = dataset.shard(num_shards, shard_index)
    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, random_seed)

    def _tokenize_lambda(line):
        utterances = tf.string_split([line], delimiter="\t").values

        srcs = [tf.string_split([utterances[t]]).values for t in range(num_inputs)]
        tgt = tf.string_split([utterances[num_inputs]]).values

        tokenized_data = {
            'tgt': tgt[:tgt_max_len] if tgt_max_len else tgt
        }

        for t in range(num_inputs):
            tokenized_data['src_%d' % t] = srcs[t][:src_max_len] if src_max_len else srcs[t]

        return tokenized_data

    src_tgt_dataset = src_tgt_dataset.map(
        _tokenize_lambda,
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    def _lookup_lambda(data):
        tgt = tf.cast(vocab_table.lookup(data['tgt']), tf.int32)
        tgt_out = tf.concat((tgt, [eos_id]), 0)
        mapped_data = {
            'tgt_in': tf.concat(([sos_id], tgt), 0),
            'tgt_out': tgt_out,
            'tgt_len': tf.size(tgt_out)
        }

        for t in range(num_inputs):
            src = tf.cast(vocab_table.lookup(data['src_%d' % t]), tf.int32)
            mapped_data['src_%d' % t] = src
            mapped_data['src_len_%d' % t] = tf.size(src)

        return mapped_data

    src_tgt_dataset = src_tgt_dataset.map(
        _lookup_lambda,
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.

    # Add in sequence lengths.
    # src_tgt_dataset = src_tgt_dataset.map(
    #     lambda srcs, tgt_in, tgt_out: (
    #         srcs, tgt_in, tgt_out,
    #         [tf.size(srcs[t]) for t in range(num_inputs)], tf.size(tgt_in)),
    #     num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    padded_shapes = {'tgt_in': tf.TensorShape([None]),
                     'tgt_out': tf.TensorShape([None]),
                     'tgt_len': tf.TensorShape([])}

    padded_values = {'tgt_in': eos_id,
                     'tgt_out': eos_id,
                     'tgt_len': 0}

    for t in range(num_inputs):
        padded_shapes['src_%d' % t] = tf.TensorShape([None])
        padded_values['src_%d' % t] = eos_id
        padded_shapes['src_len_%d' % t] = tf.TensorShape([])
        padded_values['src_len_%d' % t] = 0

    def _batching_lambda(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=padded_shapes,
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=padded_values)

    if num_buckets > 1:
        def key_func(data):
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

            bucket_id = data['tgt_len'] // bucket_width
            for t in range(num_inputs):
                bucket_id = tf.maximum(data['src_len_%d' % t] // bucket_width, bucket_id)

            # bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return _batching_lambda(windowed_data)

        batched_dataset = src_tgt_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    else:
        batched_dataset = _batching_lambda(src_tgt_dataset)

    batched_iter = batched_dataset.make_initializable_iterator()
    batched_data = batched_iter.get_next()

    return BatchedInput(
        initializer=batched_iter.initializer,
        sources=[batched_data['src_%d' % t] for t in range(num_inputs)],
        target_input=batched_data['tgt_in'],
        target_output=batched_data['tgt_out'],
        source_sequence_lengths=[batched_data['src_len_%d' % t] for t in range(num_inputs)],
        target_sequence_length=batched_data['tgt_len'])


def get_infer_iterator(test_dataset,
                       vocab_table,
                       batch_size,
                       num_turns,
                       src_max_len=None):
    num_inputs = num_turns - 1

    eos_id = tf.constant(vocab.EOS_ID, dtype=tf.int32)

    def _parse_lambda(line):
        utterances = tf.string_split([line], delimiter="\t").values
        srcs = [tf.string_split([utterances[t]]).values for t in range(num_inputs)]

        parsed_data = {}

        for t in range(num_inputs):
            src = srcs[t][:src_max_len] if src_max_len else srcs[t]
            src = tf.cast(vocab_table.lookup(src), tf.int32)
            parsed_data['src_%d' % t] = src
            parsed_data['src_len_%d' % t] = tf.size(src)

        return parsed_data

    test_dataset = test_dataset.map(_parse_lambda)

    padded_shapes = {}
    padded_values = {}

    for t in range(num_inputs):
        padded_shapes['src_%d' % t] = tf.TensorShape([None])
        padded_values['src_%d' % t] = eos_id
        padded_shapes['src_len_%d' % t] = tf.TensorShape([])
        padded_values['src_len_%d' % t] = 0

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The entry is the source line rows;
            # this has unknown-length vectors.  The last entry is
            # the source row size; this is a scalar.
            padded_shapes=padded_shapes,
            # Pad the source sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=padded_values)

    batched_dataset = batching_func(test_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()

    batched_data = batched_iter.get_next()

    return BatchedInput(
        initializer=batched_iter.initializer,
        sources=[batched_data['src_%d' % t] for t in range(num_inputs)],
        target_input=None,
        target_output=None,
        source_sequence_lengths=[batched_data['src_len_%d' % t] for t in range(num_inputs)],
        target_sequence_length=None)
