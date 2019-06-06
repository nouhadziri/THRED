import codecs
import time

import numpy as np
import tensorflow as tf

from thred.util import log
from . import bleu
from ..ncm_utils import get_translation


def decode_and_evaluate(name,
                        model,
                        sess,
                        out_file,
                        ref_file,
                        metrics,
                        beam_width,
                        num_translations_per_input=1,
                        decode=True):
    """Decode a test set and compute a score according to the evaluation task."""
    # Decode
    if decode:
        log.print_out("  decoding to output '{}'".format(out_file))

        start_time = time.time()
        num_sentences = 0
        with codecs.getwriter("utf-8")(
                tf.gfile.GFile(out_file, mode="wb")) as trans_f:
            trans_f.write("")  # Write empty string to ensure file is created.

            num_translations_per_input = max(
                min(num_translations_per_input, beam_width), 1)

            i = 0
            while True:
                i += 1
                try:

                    if i % 1000 == 0:
                        log.print_out("    decoding step {}, num sentences {}".format(i, num_sentences))

                    ncm_outputs, _ = model.decode(sess)
                    if beam_width == 0:
                        ncm_outputs = np.expand_dims(ncm_outputs, 0)

                    batch_size = ncm_outputs.shape[1]
                    num_sentences += batch_size

                    for sent_id in range(batch_size):
                        translations = [get_translation(ncm_outputs[beam_id], sent_id)
                                        for beam_id in range(num_translations_per_input)]
                        trans_f.write(b"\t".join(translations).decode("utf-8") + "\n")
                except tf.errors.OutOfRangeError:
                    log.print_time(
                        "  Done, num sentences {}, num translations per input {}".format(
                            num_sentences, num_translations_per_input), start_time)
                    break

    # Evaluation
    evaluation_scores = {}
    # if ref_file and tf.gfile.Exists(out_file):
    #     for metric in metrics:
    #         score = evaluate(ref_file, out_file, metric)
    #         evaluation_scores[metric] = score
    #         log.print_out("  %s %s: %.1f" % (metric, name, score))

    return evaluation_scores


def evaluate(ref_file, trans_file, metric):
    """Pick a metric and evaluate depending on task."""
    # BLEU scores for translation task
    if metric.lower() == "bleu":
        evaluation_score = _bleu(ref_file, trans_file)
    # ROUGE scores for summarization tasks
    elif metric.lower() == "accuracy":
        evaluation_score = _accuracy(ref_file, trans_file)
    elif metric.lower() == "word_accuracy":
        evaluation_score = _word_accuracy(ref_file, trans_file)
    else:
        raise ValueError("Unknown metric {}".format(metric))

    return evaluation_score


def _bleu(ref_file, trans_file):
    """Compute BLEU scores and handling BPE."""
    max_order = 4
    smooth = False

    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(reference_filename, "rb")) as fh:
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = reference.strip()
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)

    translations = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
        for line in fh:
            line = line.strip()
            translations.append(line.split(" "))

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = bleu.compute_bleu(
        per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score


def _accuracy(label_file, pred_file):
    """Compute accuracy, each line contains a label."""

    with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "rb")) as label_fh:
        with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as pred_fh:
            count = 0.0
            match = 0.0
            for label in label_fh:
                label = label.strip()
                pred = pred_fh.readline().strip()
                if label == pred:
                    match += 1
                count += 1
    return 100 * match / count


def _word_accuracy(label_file, pred_file):
    """Compute accuracy on per word basis."""

    with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "r")) as label_fh:
        with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "r")) as pred_fh:
            total_acc, total_count = 0., 0.
            for sentence in label_fh:
                labels = sentence.strip().split(" ")
                preds = pred_fh.readline().strip().split(" ")
                match = 0.0
                for pos in range(min(len(labels), len(preds))):
                    label = labels[pos]
                    pred = preds[pos]
                    if label == pred:
                        match += 1
                total_acc += 100 * match / max(len(labels), len(preds))
                total_count += 1
    return total_acc / total_count
