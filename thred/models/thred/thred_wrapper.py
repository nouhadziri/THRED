from os import path

import numpy as np

from thred.util import fs, log
from thred.util.embed import EmbeddingUtil
from . import thred_helper
from .. import topical_base, ncm_utils
from ..hierarchical_base import BaseHierarchicalEncoderDecoder


class TopicalHierarchicalEncoderDecoder(BaseHierarchicalEncoderDecoder):
    def __init__(self, config):
        super(TopicalHierarchicalEncoderDecoder, self).__init__(config)

    def _get_model_helper(self):
        return thred_helper

    def _get_checkpoint_name(self):
        return "thred"

    def _pre_model_creation(self):
        self.config['topic_vocab_file'] = path.join(fs.get_current_dir(self.config.vocab_file), 'topic_vocab.in')
        self._vocab_table, self.__topic_vocab_table = topical_base.initialize_vocabulary(self.config)

        EmbeddingUtil(self.config.embed_conf).build_if_not_exists(
            self.config.embedding_type, self.config.vocab_pkl, self.config.vocab_file)

        if 'original_vocab_size' not in self.config:
            self.config['original_vocab_size'] = self.config.vocab_size

        self.config.vocab_size = len(self._vocab_table)
        self.config.topic_vocab_size = len(self.__topic_vocab_table)

        if self.config.mode == "interactive" and self.config.lda_model_dir is None:
            raise ValueError("In interactive mode, THRED requires a pretrained LDA model")

    def _sample_decode(self,
                       model, global_step, sess, src_placeholder, batch_size_placeholder, eval_data, summary_writer):
        """Pick a sentence and decode."""
        decode_ids = np.random.randint(low=0, high=len(eval_data) - 1, size=1)

        sample_data = []
        for decode_id in decode_ids:
            sample_data.append(eval_data[decode_id])

        iterator_feed_dict = {
            src_placeholder: sample_data,
            batch_size_placeholder: len(decode_ids),
        }

        sess.run(model.iterator.initializer, feed_dict=iterator_feed_dict)
        ncm_outputs, infer_summary = model.decode(sess)

        for i, decode_id in enumerate(decode_ids):
            log.print_out("  # {}".format(decode_id))

            output = ncm_outputs[i]

            if self.config.beam_width > 0 and self._consider_beam():
                # get the top translation.
                output = output[0]

            translation = ncm_utils.get_translation(output, sent_id=0)
            delimited_sample = eval_data[decode_id].split("\t")
            utterances, topic = delimited_sample[:-1], delimited_sample[-1]
            sources, target = utterances[:-1], utterances[-1]

            log.print_out("    sources:")
            for t, src in enumerate(sources):
                log.print_out("      @{} {}".format(t + 1, src))
            log.print_out("    topic: {}".format(topic))
            log.print_out("    resp: {}".format(target))
            log.print_out(b"    generated: " + translation)

        # Summary
        if infer_summary is not None:
            summary_writer.add_summary(infer_summary, global_step)
