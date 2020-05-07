import codecs
import math
import os
import time
from abc import abstractmethod

import numpy as np
import tensorflow as tf

from . import ncm_utils, model_helper
from ..util import fs, misc, log, vocab
from ..util.embed import EmbeddingUtil


class AbstractModel(object):

    def init_embeddings(self, vocab_file: str, vocab_pkl: str, dtype=tf.float32, scope: str = None):
        reserved_vecs, trainable_vecs = EmbeddingUtil.load_vectors(vocab_pkl, vocab_file)

        with tf.variable_scope(scope or "embeddings", dtype=dtype):
            with tf.variable_scope(scope or "trainable_embeddings", dtype=dtype):
                reserved_token_embeddings = tf.get_variable("reserved_emb_matrix",
                                                            shape=reserved_vecs.shape,
                                                            trainable=True,
                                                            dtype=dtype)
                trainable_embeddings = tf.get_variable("trainable_emb_matrix",
                                                       shape=trainable_vecs.shape,
                                                       trainable=True,
                                                       dtype=dtype)

            self.embeddings = tf.concat([reserved_token_embeddings, trainable_embeddings], 0)

    def get_keep_probs(self, mode, params):
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            encoder_keep_prob = 1.0 - params.encoder_dropout_rate
            decoder_keep_prob = 1.0 - params.decoder_dropout_rate
        else:
            encoder_keep_prob = 1.0
            decoder_keep_prob = 1.0

        return encoder_keep_prob, decoder_keep_prob

    def _get_learning_rate_decay(self, hparams, global_step, learning_rate):
        """Get learning rate decay."""
        if hparams.learning_rate_decay_scheme == "luong10":
            start_decay_step = int(hparams.num_train_steps / 2)
            remain_steps = hparams.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / 10)  # decay 10 times
            decay_factor = 0.5
        elif hparams.learning_rate_decay_scheme == "luong234":
            start_decay_step = int(hparams.num_train_steps * 2 / 3)
            remain_steps = hparams.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / 4)  # decay 4 times
            decay_factor = 0.5
        elif hparams.learning_rate_decay_scheme == "manual":
            start_decay_step = hparams.start_decay_step
            decay_steps = hparams.decay_steps
            decay_factor = hparams.decay_factor
        else:
            start_decay_step = hparams.num_train_steps
            decay_steps = 0
            decay_factor = 1.0

        log.print_out("  learning rate decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                      "decay_factor %g" % (hparams.learning_rate_decay_scheme,
                                           start_decay_step,
                                           decay_steps,
                                           decay_factor))

        eff_global_step = global_step
        if hparams.is_pretrain_enabled():
            eff_global_step -= hparams.num_pretrain_steps

        return tf.cond(
            eff_global_step < start_decay_step,
            lambda: learning_rate,
            lambda: tf.train.exponential_decay(
                learning_rate,
                (eff_global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    def _get_sampling_probability(self, hparams, global_step, sampling_probability):
        if hparams.scheduled_sampling_decay_scheme == "luong10":
            start_decay_step = int(hparams.num_train_steps / 2)
            remain_steps = hparams.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / 10)  # decay 10 times
            decay_factor = 0.5
        elif hparams.scheduled_sampling_decay_scheme == "luong234":
            start_decay_step = int(hparams.num_train_steps * 2 / 3)
            remain_steps = hparams.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / 4)  # decay 4 times
            decay_factor = 0.5
        elif hparams.scheduled_sampling_decay_scheme == "manual":
            start_decay_step = hparams.start_decay_step
            decay_steps = hparams.decay_steps
            decay_factor = hparams.decay_factor
        else:
            start_decay_step = hparams.num_train_steps
            decay_steps = 0
            decay_factor = 1.0

        log.print_out("  scheduled sampling decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                      "decay_factor %g" % (hparams.scheduled_sampling_decay_scheme,
                                           start_decay_step,
                                           decay_steps,
                                           decay_factor))

        eff_global_step = global_step
        if hparams.is_pretrain_enabled():
            eff_global_step -= hparams.num_pretrain_steps

        return tf.cond(
            eff_global_step < start_decay_step,
            lambda: sampling_probability,
            lambda: tf.train.exponential_decay(
                sampling_probability,
                (eff_global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="sampling_prob_decay_cond")


class AbstractEncoderDecoderWrapper(object):
    def __init__(self, config) -> None:
        self.config = config

    @abstractmethod
    def train(self, target_session="", scope=None):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def interactive(self, scope=None):
        pass


class NMTEncoderDecoderWrapper(AbstractEncoderDecoderWrapper):
    def __init__(self, config) -> None:
        super(NMTEncoderDecoderWrapper, self).__init__(config)
        self.config['metrics'] = ['dev_ppl']
        self.config.metrics.extend(self._get_metrics())

        for metric in self.config.metrics:
            self.config['best_{}_dir'.format(metric)] = os.path.join(self.config.model_dir, 'best_{}'.format(metric))
            fs.mkdir_if_not_exists(self.config['best_{}_dir'.format(metric)])
            best_metric = 'best_{}'.format(metric)
            if best_metric not in self.config:
                self.config[best_metric] = float('inf')

        self.config['checkpoint_file'] = os.path.join(self.config.model_dir,
                                                      '{}.ckpt'.format(self._get_checkpoint_name()))

        if self.config.mode == 'train':
            self.config['num_train_steps'] = int(self.config.num_train_epochs * math.ceil(
                fs.count_lines(self.config.train_data) / self.config.batch_size))

        self.config.vocab_file = os.path.join(self.config.model_dir,
                                              'vocab{}.in'.format(self.config.original_vocab_size
                                                                  if 'original_vocab_size' in self.config
                                                                  else self.config.vocab_size))

        self.config.vocab_pkl = os.path.join(self.config.model_dir,
                                             'vocab_{}.pkl'.format(self.config.embedding_type))

        if 'epoch_step' not in self.config:
            self.config['epoch_step'] = 0

        if 'epoch' not in self.config:
            self.config['epoch'] = 0

        self._vocab_table = None

    def _get_metrics(self):
        return []

    def _get_model_helper(self):
        raise NotImplementedError()

    def _get_checkpoint_name(self):
        raise NotImplementedError()

    def _consider_beam(self):
        return True

    def _pre_model_creation(self):
        vocab.create_vocabulary(self.config.vocab_file, self.config.train_data, self.config.vocab_size)
        EmbeddingUtil(self.config.embed_conf).build_if_not_exists(
            self.config.embedding_type, self.config.vocab_pkl, self.config.vocab_file)

        self._vocab_table = vocab.create_vocab_dict(self.config.vocab_file)
        if self.config.vocab_size > len(self._vocab_table):
            if 'original_vocab_size' not in self.config:
                self.config['original_vocab_size'] = self.config.vocab_size
            self.config.vocab_size = len(self._vocab_table)
            print('  [WARN] vocab size decreased to {}'.format(self.config.vocab_size))

    def _post_model_creation(self, train_model, eval_model, infer_model):
        pass

    def _sample_decode(self,
                       model, global_step, sess, src_placeholder, batch_size_placeholder, eval_data, summary_writer):
        pass

    def _format_results(self, name, ppl, scores, metrics):
        """Format results."""
        result_str = "%s ppl %.2f" % (name, ppl)
        if scores:
            for metric in metrics:
                result_str += ", %s %s %.1f" % (name, metric, scores[metric])
        return result_str

    def _get_best_results(self):
        """Summary of the current best results."""
        tokens = []
        for metric in self.config.metrics:
            tokens.append("%s %.2f" % (metric, getattr(self.config, "best_" + metric)))
        return ", ".join(tokens)

    def init_stats(self):
        """Initialize statistics that we want to keep."""

        return {"step_time": 0.0, "loss": 0.0, "predict_count": 0.0,
                "total_count": 0.0, "grad_norm": 0.0}

    def update_stats(self, stats, summary_writer, start_time, step_result):
        """Update stats: write summary and accumulate statistics."""
        (_, step_loss, step_predict_count, step_summary, global_step,
         step_word_count, batch_size, grad_norm, learning_rate) = step_result

        # Write step summary.
        summary_writer.add_summary(step_summary, global_step)

        # update statistics
        stats["step_time"] += (time.time() - start_time)
        stats["loss"] += (step_loss * batch_size)
        stats["predict_count"] += step_predict_count
        stats["total_count"] += float(step_word_count)
        stats["grad_norm"] += grad_norm
        stats["learning_rate"] = learning_rate

        return global_step

    def check_stats(self, stats, global_step, steps_per_stats, log_f, pbar=None):
        """Print statistics and also check for overflow."""
        # Print statistics for the previous epoch.
        avg_step_time = stats["step_time"] / steps_per_stats
        avg_grad_norm = stats["grad_norm"] / steps_per_stats
        train_ppl = misc.safe_exp(
            stats["loss"] / stats["predict_count"])
        speed = stats["total_count"] / (1000 * stats["step_time"])

        if pbar:
            pbar.set_postfix(lr=stats["learning_rate"],
                             wps='{:.1f}K'.format(speed),
                             ppl='{:.3f}'.format(train_ppl),
                             best_dev_ppl='{:.3f}'.format(self.config.best_dev_ppl),
                             gN='{:.2f}'.format(avg_grad_norm))

        log.print_out(
            "  global step %d lr %g "
            "step-time %.2fs wps %.2fK ppl %.2f gN %.2f" %
            (global_step, stats["learning_rate"],
             avg_step_time, speed, train_ppl, avg_grad_norm),
            log_f, skip_stdout=True)

        # Check for overflow
        is_overflow = False
        if math.isnan(train_ppl) or math.isinf(train_ppl) or train_ppl > 1e20:
            log.print_out("  step %d overflow, stop early" % global_step, log_f)
            is_overflow = True

        return train_ppl, speed, is_overflow

    def _load_data(self, input_file):
        """Load inference data."""
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(input_file, mode="rb")) as f:
            inference_data = [line.strip() for line in f.readlines()]

        return inference_data

    def _decode_and_evaluate(self,
                             model, infer_sess, iterator_feed_dict,
                             num_responses_per_input=1, label="tests"):
        start_time = time.time()
        num_sentences = 0

        out_file = os.path.join(self.config.model_dir, "output_%s" % label)

        _beam_width = self.config.beam_width if self._consider_beam() else 0

        num_responses_per_input = max(
            min(num_responses_per_input, _beam_width), 1)

        infer_sess.run(model.iterator.initializer, feed_dict=iterator_feed_dict)

        with codecs.getwriter("utf-8")(
                tf.gfile.GFile(out_file, mode="wb")) as trans_f:
            trans_f.write("")  # Write empty string to ensure file is created.

            while True:
                try:
                    ncm_outputs, _ = model.decode(infer_sess)
                    if self.config.beam_width == 0 or not self._consider_beam():
                        ncm_outputs = np.expand_dims(ncm_outputs, 0)

                    batch_size = ncm_outputs.shape[1]
                    num_sentences += batch_size

                    for sent_id in range(batch_size):
                        responses = [ncm_utils.get_translation(ncm_outputs[beam_id], sent_id)
                                     for beam_id in range(num_responses_per_input)]
                        trans_f.write(b"\t".join(responses).decode("utf-8") + "\n")
                except tf.errors.OutOfRangeError:
                    break

        log.print_time(
            "  Done, num sentences %d, num translations per input %d" %
            (num_sentences, num_responses_per_input), start_time)

    def run_internal_eval(self, eval_model, eval_sess, model_dir, summary_writer):
        """Compute internal evaluation (perplexity) for both dev / tests."""
        with eval_model.graph.as_default():
            loaded_eval_model, global_step = model_helper.create_or_load_model(
                eval_model.model, model_dir, eval_sess, "eval")

        dev_eval_iterator_feed_dict = {
            eval_model.eval_file_placeholder: self.config.dev_data
        }

        eval_sess.run(eval_model.iterator.initializer, feed_dict=dev_eval_iterator_feed_dict)
        dev_ppl = model_helper.compute_perplexity(loaded_eval_model, eval_sess, "dev", self.config.dev_size)
        log.add_summary(summary_writer, global_step, "dev_ppl", dev_ppl)

        if dev_ppl < self.config.best_dev_ppl:
            loaded_eval_model.saver.save(eval_sess,
                                         os.path.join(self.config.best_dev_ppl_dir,
                                                      '{}.ckpt'.format(self._get_checkpoint_name())),
                                         global_step=global_step)

        return dev_ppl

    def run_infer_best_model(self, infer_model, eval_model, infer_sess, eval_sess, best_model_dir, label,
                             summary_writer):
        with eval_model.graph.as_default():
            loaded_eval_model, global_step = model_helper.create_or_load_model(
                eval_model.model, best_model_dir, eval_sess, "eval")

        infer_eval_iterator_feed_dict = {
            eval_model.eval_file_placeholder: self.config.test_data
        }

        eval_sess.run(eval_model.iterator.initializer, feed_dict=infer_eval_iterator_feed_dict)
        test_ppl = model_helper.compute_perplexity(loaded_eval_model, eval_sess, "test")
        log.add_summary(summary_writer, global_step, "test_ppl", test_ppl)

        with infer_model.graph.as_default():
            loaded_infer_model, global_step = model_helper.create_or_load_model(
                infer_model.model, best_model_dir, infer_sess, "infer")

        infer_feed_dict = {
            infer_model.src_placeholder: self._load_data(self.config.test_data),
            infer_model.batch_size_placeholder: self.config.infer_batch_size,
        }
        self._decode_and_evaluate(loaded_infer_model,
                                  infer_sess, infer_feed_dict, label=label)
        return test_ppl

    def run_sample_decode(self, infer_model, infer_sess, model_dir, summary_writer, eval_data):
        """Sample decode a random sentence from src_data."""
        with infer_model.graph.as_default():
            loaded_infer_model, global_step = model_helper.create_or_load_model(
                infer_model.model, model_dir, infer_sess, "infer")

        self._sample_decode(loaded_infer_model, global_step, infer_sess,
                            infer_model.src_placeholder, infer_model.batch_size_placeholder,
                            eval_data, summary_writer)

    def interactive(self, scope=None):
        import platform
        from prompt_toolkit import prompt
        from prompt_toolkit.history import FileHistory
        from thred.topic_model.lda import TopicInferer
        from thred.util.nlp import NLPToolkit

        nlp_toolkit = NLPToolkit()

        __os = platform.system()

        self.config.infer_batch_size = 1
        self._pre_model_creation()

        if self.config.lda_model_dir is not None:
            topic_inferer = TopicInferer(self.config.lda_model_dir)
        else:
            topic_inferer = None

        infer_model = self._get_model_helper().create_infer_model(self.config, scope)

        with tf.Session(
                config=model_helper.get_config_proto(self.config.log_device), graph=infer_model.graph) as sess:
            latest_ckpt = tf.train.latest_checkpoint(self.config.get_infer_model_dir())
            loaded_infer_model = model_helper.load_model(
                infer_model.model, latest_ckpt, sess, "infer")

            log.print_out("# Start decoding")

            if __os == 'Windows':
                sentence = input("> ").strip()
            else:
                sentence = prompt(">>> ",
                                  history=FileHistory(os.path.join(self.config.model_dir, ".chat_history"))).strip()
            conversation = [vocab.EOS] * (self.config.num_turns - 1)

            while sentence:
                current_utterance = ' '.join(nlp_toolkit.tokenize(sentence)).lower()
                conversation.append(current_utterance)
                conversation.pop(0)

                feedable_context = "\t".join(conversation)
                if topic_inferer is None:
                    iterator_feed_dict = {
                        infer_model.src_placeholder: [feedable_context],
                        infer_model.batch_size_placeholder: 1,
                    }
                else:
                    _, topic_words = topic_inferer.from_collection(
                        [feedable_context],
                        dialogue_as_doc=True,
                        words_per_topic=self.config.topic_words_per_utterance)[0]
                    iterator_feed_dict = {
                        infer_model.src_placeholder:
                            [feedable_context + "\t" + " ".join(topic_words)],
                        infer_model.batch_size_placeholder: 1,
                    }

                sess.run(infer_model.iterator.initializer, feed_dict=iterator_feed_dict)
                output, _ = loaded_infer_model.decode(sess)
                if self.config.beam_width > 0 and self._consider_beam():
                    # get the top translation.
                    output = output[0]

                resp = ncm_utils.get_translation(output, sent_id=0)

                log.print_out(resp + b"\n")

                if __os == 'Windows':
                    sentence = input("> ").strip()
                else:
                    sentence = prompt(">>> ",
                                      history=FileHistory(os.path.join(self.config.model_dir, ".chat_history"))).strip()

        print("Bye!!!")
