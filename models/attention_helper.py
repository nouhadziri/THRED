import tensorflow as tf


def is_attention_enabled(attention_option):
    return attention_option is not None and \
           attention_option in ("luong", "scaled_luong", "bahdanau", "normed_bahdanau")


def create_attention_mechanism(attention_option, num_units, memory, memory_length):
    """Create attention mechanism based on the attention_option."""

    # Mechanism
    if attention_option == "luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=memory_length)
    elif attention_option == "scaled_luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units,
            memory,
            memory_sequence_length=memory_length,
            scale=True)
    elif attention_option == "bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=memory_length)
    elif attention_option == "normed_bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units,
            memory,
            memory_sequence_length=memory_length,
            normalize=True)
    else:
        raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism