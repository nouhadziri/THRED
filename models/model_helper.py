import collections
import time

import tensorflow as tf

from util import log, misc


class TopicalBatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "sources",
                            "topic", "target_input", "target_output",
                            "source_sequence_lengths",
                            "topic_sequence_length", "target_sequence_length"))):
    pass


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "sources", "target_input",
                            "target_output", "source_sequence_lengths",
                            "target_sequence_length"))):
    pass


class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator",
                                          "skip_count_placeholder"))):
    pass


class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model",
                            "eval_file_placeholder", "iterator"))):
    pass


class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model", "src_placeholder",
                            "batch_size_placeholder", "iterator"))):
    pass


def get_config_proto(log_device_placement):
    return tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True))


def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    log.print_out(
        "  loaded %s model parameters from %s, time %.2fs" %
        (name, ckpt, time.time() - start_time))
    return model


def create_or_load_model(model, model_dir, session, name):
    """Create a model and initialize or load parameters in session."""

    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        log.print_out("  created %s model with fresh parameters, time %.2fs" %
                      (name, time.time() - start_time))

    global_step = model.global_step.eval(session=session)
    return model, global_step


def compute_perplexity(model, sess, name):
    """Compute perplexity of the output of the model.
    Args:
      model: model for compute perplexity.
      sess: tensorflow session to use.
      name: name of the batch.
    Returns:
      The perplexity of the eval outputs.
    """
    total_loss = 0
    total_predict_count = 0
    start_time = time.time()
    step = 0

    while True:
        try:
            loss, predict_count, batch_size = model.eval(sess)
            total_loss += loss * batch_size
            total_predict_count += predict_count
            step += 1
            if step % 500 == 0:
                ls = total_loss / total_predict_count
                ppl = misc.safe_exp(ls)
                log.print_out("    ## After %d steps, loss %.2f - ppl %.3f" % (step, ls, ppl))
        except tf.errors.OutOfRangeError:
            break

    perplexity = misc.safe_exp(total_loss / total_predict_count)
    log.print_time("  eval %s: perplexity %.2f" % (name, perplexity), start_time)
    return perplexity
