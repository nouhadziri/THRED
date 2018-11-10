import codecs
from os import path

import tensorflow as tf
import numpy as np

from util import fs, log, vocab
from util.embed import WordEmbeddings
from models.hierarchical_base import BaseHierarchicalEncoderDecoder
from models import topical_base, ncm_utils
from models.thred import thred_helper


class TopicalHierarchicalEncoderDecoder(BaseHierarchicalEncoderDecoder):
    def __init__(self, config):
        super(TopicalHierarchicalEncoderDecoder, self).__init__(config)

    def _get_model_helper(self):
        return thred_helper

    def _get_checkpoint_name(self):
        return "thred"

    def _pre_model_creation(self):
        self.config['topic_vocab_file'] = path.join(fs.split3(self.config.vocab_file)[0], 'topic_vocab.in')
        self._vocab_table, self.__topic_vocab_table = topical_base.initialize_vocabulary(self.config)

        WordEmbeddings(self.config.embed_conf).create_and_save(
            self.config.vocab_pkl, self.config.vocab_file,
            self.config.embedding_type, self.config.embedding_size)

        if 'original_vocab_size' not in self.config:
            self.config['original_vocab_size'] = self.config.vocab_size

        self.config.vocab_size = len(self._vocab_table)
        self.config.topic_vocab_size = len(self.__topic_vocab_table)

        if self.config.mode == "interactive" and self.config.lda_model_dir is None:
            raise ValueError("In interactive mode, THRED requires a pretrained LDA model")

    def _load_data(self, input_file, include_target=False):
        """Load inference data."""
        inference_data = []
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(input_file, mode="rb")) as f:
            for line in f:
                utterances_str, topic_str = tuple(line.split(topical_base.DEFAULT_SEPARATOR))
                utterances, topic = utterances_str.strip().split("\t"), topic_str.strip()
                sources = "\t".join(utterances[:-1]) + topical_base.DEFAULT_SEPARATOR + topic
                if include_target:
                    inference_data.append((sources, utterances[-1]))
                else:
                    inference_data.append(sources)

        return inference_data

    def _sample_decode(self,
                       model, global_step, sess, src_placeholder, batch_size_placeholder, eval_data, summary_writer):
        """Pick a sentence and decode."""
        decode_ids = np.random.randint(low=0, high=len(eval_data) - 1, size=1)

        sample_data = []
        for decode_id in decode_ids:
            sample_data.append(eval_data[decode_id][0])

        iterator_feed_dict = {
            src_placeholder: sample_data,
            batch_size_placeholder: len(decode_ids),
        }

        sess.run(model.iterator.initializer, feed_dict=iterator_feed_dict)
        ncm_outputs, infer_summary = model.decode(sess)

        for i, decode_id in enumerate(decode_ids):
            log.print_out("  # %d" % decode_id)

            output = ncm_outputs[i]

            if self.config.beam_width > 0 and self._consider_beam():
                # get the top translation.
                output = output[0]

            translation = ncm_utils.get_translation(output, sent_id=0)
            log.print_out("    sources:")
            sources, topic = tuple(eval_data[decode_id][0].split(topical_base.DEFAULT_SEPARATOR))
            for t, src in enumerate(sources.split("\t")):
                log.print_out("      @%d %s" % (t + 1, src))
            log.print_out("    topic: %s" % topic)
            log.print_out("    resp: %s" % eval_data[decode_id][1])
            log.print_out(b"    generated: " + translation)

        # Summary
        if infer_summary is not None:
            summary_writer.add_summary(infer_summary, global_step)

