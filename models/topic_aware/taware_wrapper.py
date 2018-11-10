import math
import os
import random
import time
import codecs

import tensorflow as tf

import models.model_helper
from models import model_helper, ncm_utils, topical_base
from models.base import NMTEncoderDecoderWrapper
from models.topic_aware import taware_helper
from models.topic_aware import taware_model
from models.vanilla import eval_metric
from topic_model.lda import DEFAULT_SEPARATOR
from util import log, misc, fs
from util.misc import Stopwatch
from util.embed import WordEmbeddings


class TopicAwareNMTEncoderDecoder(NMTEncoderDecoderWrapper):
    def __init__(self, config) -> None:
        if config.mode != 'train' and 'original_vocab_size' in config:
            config.vocab_size = config.original_vocab_size

        super(TopicAwareNMTEncoderDecoder, self).__init__(config)

    def _get_checkpoint_name(self):
        return 'taware'

    def _pre_model_creation(self):
        self.config['topic_vocab_file'] = os.path.join(fs.split3(self.config.vocab_file)[0], 'topic_vocab.in')
        self._vocab_table, self.__topic_vocab_table = topical_base.initialize_vocabulary(self.config)

        WordEmbeddings(self.config.embed_conf).create_and_save(
            self.config.vocab_pkl, self.config.vocab_file,
            self.config.embedding_type, self.config.embedding_size)

        if 'original_vocab_size' not in self.config:
            self.config['original_vocab_size'] = self.config.vocab_size
        self.config.vocab_size = len(self._vocab_table)
        self.config.topic_vocab_size = len(self.__topic_vocab_table)

        if self.config.mode == "interactive" and self.config.lda_model_dir is None:
            raise ValueError("In interactive mode, TA-Seq2Seq requires a pretrained LDA model")

    def run_sample_decode(self, infer_model, infer_sess, model_dir, summary_writer, eval_data):
        """Sample decode a random sentence from src_data."""
        with infer_model.graph.as_default():
            loaded_infer_model, global_step = model_helper.create_or_load_model(
                infer_model.model, model_dir, infer_sess, "infer")

        self.__sample_decode(loaded_infer_model, global_step, infer_sess,
                            infer_model.iterator, eval_data,
                            infer_model.src_placeholder,
                            infer_model.batch_size_placeholder, summary_writer)

    def run_internal_eval(self,
                          eval_model, eval_sess, model_dir, summary_writer, use_test_set=True):
        """Compute internal evaluation (perplexity) for both dev / test."""
        with eval_model.graph.as_default():
            loaded_eval_model, global_step = model_helper.create_or_load_model(
                eval_model.model, model_dir, eval_sess, "eval")

        dev_file = self.config.dev_data

        dev_eval_iterator_feed_dict = {
            eval_model.eval_file_placeholder: dev_file
        }

        dev_ppl = self._internal_eval(loaded_eval_model, global_step, eval_sess,
                                      eval_model.iterator, dev_eval_iterator_feed_dict,
                                      summary_writer, "dev")
        log.add_summary(summary_writer, global_step, "dev_ppl", dev_ppl)

        if dev_ppl < self.config.best_dev_ppl:
            loaded_eval_model.saver.save(eval_sess,
                                         os.path.join(self.config.best_dev_ppl_dir, 'taware.ckpt'),
                                         global_step=global_step)

        test_ppl = None
        if use_test_set:
            test_file = self.config.test_data

            test_eval_iterator_feed_dict = {
                eval_model.eval_file_placeholder: test_file
            }
            test_ppl = self._internal_eval(loaded_eval_model, global_step, eval_sess,
                                           eval_model.iterator, test_eval_iterator_feed_dict,
                                           summary_writer, "test")
        return dev_ppl, test_ppl

    def run_external_eval(self,
                          infer_model, infer_sess, model_dir,
                          summary_writer, save_best_dev=True, use_test_set=True):

        """Compute external evaluation (bleu, rouge, etc.) for both dev / test."""
        with infer_model.graph.as_default():
            loaded_infer_model, global_step = model_helper.create_or_load_model(
                infer_model.model, model_dir, infer_sess, "infer")

        dev_infer_iterator_feed_dict = {
            infer_model.src_placeholder: self._load_data(self.config.dev_data),
            infer_model.batch_size_placeholder: self.config.infer_batch_size,
        }

        dev_scores = self._external_eval(
            loaded_infer_model,
            global_step,
            infer_sess,
            infer_model.iterator,
            dev_infer_iterator_feed_dict,
            self.config.dev_data,
            "dev",
            summary_writer,
            save_on_best=save_best_dev)

        test_scores = None
        if use_test_set:
            test_file = self.config.test_data
            test_infer_iterator_feed_dict = {
                infer_model.src_placeholder: self._load_data(test_file),
                infer_model.batch_size_placeholder: self.config.infer_batch_size,
            }

            test_scores = self._external_eval(
                loaded_infer_model,
                global_step,
                infer_sess,
                infer_model.iterator,
                test_infer_iterator_feed_dict,
                test_file,
                "test",
                summary_writer,
                save_on_best=False)
        return dev_scores, test_scores, global_step

    def run_full_eval(self, model_dir, infer_model, infer_sess, eval_model, eval_sess, summary_writer, eval_data):
        """Wrapper for running sample_decode, internal_eval and external_eval."""
        self.run_sample_decode(infer_model, infer_sess, model_dir, summary_writer, eval_data)
        dev_ppl, test_ppl = self.run_internal_eval(eval_model, eval_sess, model_dir, summary_writer)
        dev_scores, test_scores, global_step = self.run_external_eval(
            infer_model, infer_sess, model_dir, summary_writer)

        result_summary = self._format_results("dev", dev_ppl, dev_scores, self.config.metrics)
        result_summary += ", " + self._format_results("test", test_ppl, test_scores, self.config.metrics)

        return result_summary, global_step, dev_scores, test_scores, dev_ppl, test_ppl

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

    def check_stats(self, stats, global_step, steps_per_stats, log_f):
        """Print statistics and also check for overflow."""
        # Print statistics for the previous epoch.
        avg_step_time = stats["step_time"] / steps_per_stats
        avg_grad_norm = stats["grad_norm"] / steps_per_stats
        train_ppl = misc.safe_exp(
            stats["loss"] / stats["predict_count"])
        speed = stats["total_count"] / (1000 * stats["step_time"])
        log.print_out(
            "  global step %d lr %g "
            "step-time %.2fs wps %.2fK ppl %.2f gN %.2f %s" %
            (global_step, stats["learning_rate"],
             avg_step_time, speed, train_ppl, avg_grad_norm,
             self._get_best_results()),
            log_f)

        # Check for overflow
        is_overflow = False
        if math.isnan(train_ppl) or math.isinf(train_ppl) or train_ppl > 1e20:
            log.print_out("  step %d overflow, stop early" % global_step, log_f)
            is_overflow = True

        return train_ppl, speed, is_overflow

    def _load_data(self, input_file, include_target=False):
        """Load inference data."""

        inference_data = []
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(input_file, mode="rb")) as f:
            for line in f:
                utterances_str, topics_str = tuple(line.split(DEFAULT_SEPARATOR))
                utterances, topics = utterances_str.strip().split('\t'), topics_str.strip().split()
                inference_data.append((utterances, topics))

        return inference_data

    def train(self, target_session="", scope=None):
        out_dir = self.config.model_dir
        model_dir = out_dir

        num_train_steps = self.config.num_train_steps
        steps_per_stats = self.config.steps_per_stats
        # steps_per_external_eval = self.config.steps_per_external_eval
        steps_per_eval = 20 * steps_per_stats
        # if not steps_per_external_eval:
        #     steps_per_external_eval = 5 * steps_per_eval

        self._pre_model_creation()

        train_model = taware_helper.create_train_model(taware_model.TopicAwareSeq2SeqModel, self.config, scope)
        eval_model = taware_helper.create_eval_model(taware_model.TopicAwareSeq2SeqModel, self.config, scope)
        infer_model = taware_helper.create_infer_model(taware_model.TopicAwareSeq2SeqModel, self.config, scope)

        # Preload data for sample decoding.
        dev_file = self.config.dev_data
        eval_data = self._load_data(dev_file, include_target=True)

        summary_name = "train_log"

        # Log and output files
        log_file = os.path.join(out_dir, "log_%d" % time.time())
        log_f = tf.gfile.GFile(log_file, mode="a")
        log.print_out("# log_file=%s" % log_file, log_f)

        avg_step_time = 0.0

        # TensorFlow model
        config_proto = models.model_helper.get_config_proto(self.config.log_device)

        train_sess = tf.Session(
            target=target_session, config=config_proto, graph=train_model.graph)
        eval_sess = tf.Session(
            target=target_session, config=config_proto, graph=eval_model.graph)
        infer_sess = tf.Session(
            target=target_session, config=config_proto, graph=infer_model.graph)

        with train_model.graph.as_default():
            loaded_train_model, global_step = model_helper.create_or_load_model(
                train_model.model, model_dir, train_sess, "train")

        # Summary writer
        summary_writer = tf.summary.FileWriter(
            os.path.join(out_dir, summary_name), train_model.graph)

        # First evaluation
        # self.run_full_eval(
        #    model_dir, infer_model, infer_sess,
        #    eval_model, eval_sess, summary_writer, eval_data)

        last_stats_step = global_step
        last_eval_step = global_step
        # last_external_eval_step = global_step
        patience = self.config.patience

        # This is the training loop.
        stats = self.init_stats()
        speed, train_ppl = 0.0, 0.0
        start_train_time = time.time()

        log.print_out(
            "# Start step %d, epoch %d, lr %g, %s" %
            (global_step, self.config.epoch, loaded_train_model.learning_rate.eval(session=train_sess),
             time.ctime()),
            log_f)

        self.config.save()
        log.print_out("# Configs saved")

        # Initialize all of the iterators
        skip_count = self.config.batch_size * self.config.epoch_step
        log.print_out("# Init train iterator for %d steps, skipping %d elements" %
                      (self.config.num_train_steps, skip_count))

        train_sess.run(
            train_model.iterator.initializer,
            feed_dict={train_model.skip_count_placeholder: skip_count})

        while self.config.epoch < self.config.num_train_epochs and patience > 0:
            ### Run a step ###
            start_time = time.time()
            try:
                step_result = loaded_train_model.train(train_sess)
                self.config.epoch_step += 1
            except tf.errors.OutOfRangeError:
                # Finished going through the training dataset.  Go to next epoch.
                sw = Stopwatch()
                log.print_out(
                    "# Finished an epoch, step %d. Perform external evaluation" %
                    global_step)
                self.run_sample_decode(infer_model, infer_sess,
                                       model_dir, summary_writer, eval_data)

                log.print_out(
                    "## Done epoch %d in %d steps. step %d @ eval time: %ds" %
                    (self.config.epoch, self.config.epoch_step, global_step, sw.elapsed()))

                self.config.epoch += 1
                self.config.epoch_step = 0
                self.config.save()

                train_sess.run(
                    train_model.iterator.initializer,
                    feed_dict={train_model.skip_count_placeholder: 0})
                continue

            # Write step summary and accumulate statistics
            global_step = self.update_stats(stats, summary_writer, start_time, step_result)

            # Once in a while, we print statistics.
            if global_step - last_stats_step >= steps_per_stats:
                last_stats_step = global_step
                train_ppl, speed, is_overflow = self.check_stats(stats, global_step, steps_per_stats, log_f)
                if is_overflow:
                    break

                # Reset statistics
                stats = self.init_stats()

            if global_step - last_eval_step >= steps_per_eval:
                last_eval_step = global_step

                log.print_out("# Save eval, global step %d" % global_step)
                log.add_summary(summary_writer, global_step, "train_ppl", train_ppl)

                # Save checkpoint
                loaded_train_model.saver.save(
                    train_sess,
                    self.config.checkpoint_file,
                    global_step=global_step)

                # Evaluate on dev
                self.run_sample_decode(infer_model, infer_sess, model_dir, summary_writer, eval_data)
                dev_ppl, _ = self.run_internal_eval(eval_model, eval_sess, model_dir, summary_writer, use_test_set=False)

                if dev_ppl < self.config.best_dev_ppl:
                    self.config.best_dev_ppl = dev_ppl
                    patience = self.config.patience
                    log.print_out('    ** Best model thus far, ep {}|{} dev_ppl {:.3f}'.format(
                        self.config.epoch,
                        self.config.epoch_step,
                        dev_ppl))
                elif dev_ppl > self.config.degrade_threshold * self.config.best_dev_ppl:
                    patience -= 1
                    log.print_out(
                        '    worsened, ep {}|{} patience {} best_dev_ppl {:.3f}'.format(
                            self.config.epoch,
                            self.config.epoch_step,
                            self.config.patience,
                            self.config.best_dev_ppl))

                # Save config parameters
                self.config.save()

            # if global_step - last_external_eval_step >= steps_per_external_eval:
            #     last_external_eval_step = global_step
            #
            #     # Save checkpoint
            #     loaded_train_model.saver.save(
            #         train_sess,
            #         self.config.checkpoint_file,
            #         global_step=global_step)
            #     self.run_sample_decode(infer_model, infer_sess,
            #                            model_dir, summary_writer, eval_data)
                # dev_scores, test_scores, _ = self.run_external_eval(infer_model, infer_sess, model_dir, summary_writer)

        # Done training
        loaded_train_model.saver.save(
            train_sess,
            self.config.checkpoint_file,
            global_step=global_step)

        # result_summary, _, dev_scores, test_scores, dev_ppl, test_ppl = self.run_full_eval(
        #     model_dir, infer_model, infer_sess,
        #     eval_model, eval_sess,
        #     summary_writer, eval_data)
        dev_scores, test_scores, dev_ppl, test_ppl = None, None, None, None
        result_summary = ""

        log.print_out(
            "# Final, step %d lr %g "
            "step-time %.2f wps %.2fK ppl %.2f, %s, %s" %
            (global_step, loaded_train_model.learning_rate.eval(session=train_sess),
             avg_step_time, speed, train_ppl, result_summary, time.ctime()),
            log_f)
        log.print_time("# Done training!", start_train_time)

        summary_writer.close()

        # log.print_out("# Start evaluating saved best models.")
        # for metric in self.config.metrics:
        #     best_model_dir = getattr(self.config, "best_" + metric + "_dir")
        #     summary_writer = tf.summary.FileWriter(
        #         os.path.join(best_model_dir, summary_name), infer_model.graph)
        #     result_summary, best_global_step, _, _, _, _ = self.run_full_eval(
        #         best_model_dir, infer_model, infer_sess, eval_model, eval_sess,
        #         summary_writer, eval_data)
        #     log.print_out("# Best %s, step %d "
        #                   "step-time %.2f wps %.2fK, %s, %s" %
        #                   (metric, best_global_step, avg_step_time, speed,
        #                    result_summary, time.ctime()), log_f)
        #     summary_writer.close()

        return (dev_scores, test_scores, dev_ppl, test_ppl, global_step)

    def __sample_decode(self,
                       model, global_step, sess, iterator, eval_data,
                       iterator_src_placeholder,
                       iterator_batch_size_placeholder, summary_writer):
        """Pick a sentence and decode."""
        decode_id = random.randint(0, len(eval_data) - 1)
        log.print_out("  # {}".format(decode_id))

        sample_utterances, sample_topic_words = eval_data[decode_id]
        sample_topic_words = " ".join(sample_topic_words)

        iterator_feed_dict = {
            iterator_src_placeholder: [DEFAULT_SEPARATOR.join(["\t".join(sample_utterances), sample_topic_words])],
            iterator_batch_size_placeholder: 1,
        }
        sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

        ncm_outputs, attention_summary = model.decode(sess)

        if self.config.beam_width > 0:
            # get the top translation.
            ncm_outputs = ncm_outputs[0]

        translation = ncm_utils.get_translation(ncm_outputs, sent_id=0)
        log.print_out("    sources:")
        for t, src in enumerate(sample_utterances[:-1]):
            log.print_out("      @{} {}".format(t + 1, src))
        log.print_out("    topicals: {}".format(sample_topic_words))
        log.print_out("    resp: {}".format(sample_utterances[-1]))
        log.print_out(b"    ncm: " + translation)

        # Summary
        if attention_summary is not None:
            summary_writer.add_summary(attention_summary, global_step)

    def _internal_eval(self,
                       model, global_step, sess, iterator, iterator_feed_dict,
                       summary_writer, label):
        """Computing perplexity."""
        sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
        ppl = model_helper.compute_perplexity(model, sess, label)
        log.add_summary(summary_writer, global_step, "%s_ppl" % label, ppl)
        return ppl

    def _external_eval(self,
                       model, global_step, sess, iterator,
                       iterator_feed_dict, eval_file, label, summary_writer,
                       save_on_best):
        """External evaluation such as BLEU and ROUGE scores."""
        out_dir = self.config.model_dir
        decode = global_step > 0
        if decode:
            log.print_out("# External evaluation, global step %d" % global_step)

        sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

        output = os.path.join(out_dir, "output_%s" % label)
        scores = eval_metric.decode_and_evaluate(
            label,
            model,
            sess,
            output,
            ref_file=eval_file,
            metrics=self.config.metrics,
            beam_width=self.config.beam_width,
            decode=decode)
        # Save on best metrics
        if decode:
            for metric in self.config.metrics:
                log.add_summary(summary_writer, global_step, "%s_%s" % (label, metric), scores[metric])
                # metric: larger is better
                if save_on_best and scores[metric] > getattr(self.config, "best_" + metric):
                    setattr(self.config, "best_" + metric, scores[metric])
                    model.saver.save(
                        sess,
                        os.path.join(
                            getattr(self.config, "best_" + metric + "_dir"), self._get_checkpoint_name() + ".ckpt"),
                        global_step=model.global_step)
            # self.config.save(out_dir)
        return scores

    def interactive(self):
        from prompt_toolkit import prompt
        from prompt_toolkit.history import FileHistory
        from topic_model.lda import TopicInferer, DEFAULT_SEPARATOR
        from util.nlp import NLPToolkit

        nlp_toolkit = NLPToolkit()

        self._pre_model_creation()
        topic_inferer = TopicInferer(self.config.lda_model_dir)

        infer_model = taware_helper.create_infer_model(taware_model.TopicAwareSeq2SeqModel, self.config)
        config_proto = models.model_helper.get_config_proto(self.config.log_device)

        with tf.Session(graph=infer_model.graph, config=config_proto) as sess:
            ckpt = tf.train.latest_checkpoint(self.config.model_dir)
            loaded_infer_model = model_helper.load_model(
                infer_model.model, ckpt, sess, "infer")

            log.print_out("# Start decoding")

            sentence = prompt(">>> ", history=FileHistory(os.path.join(self.config.model_dir, ".chat_history"))).strip()

            while sentence:
                utterance = ' '.join(nlp_toolkit.tokenize(sentence)).lower()
                topic_words = topic_inferer.from_collection([utterance],
                                                            dialogue_as_doc=True,
                                                            words_per_topic=self.config.topic_words_per_utterance)

                iterator_feed_dict = {
                    infer_model.src_placeholder: [utterance + DEFAULT_SEPARATOR + " ".join(topic_words)],
                    infer_model.batch_size_placeholder: 1,
                }
                sess.run(infer_model.iterator.initializer, feed_dict=iterator_feed_dict)
                output, _ = loaded_infer_model.decode(sess)

                if self.config.beam_width > 0:
                    # get the top translation.
                    output = output[0]

                resp = ncm_utils.get_translation(output, sent_id=0)

                log.print_out(resp + b"\n")

                sentence = prompt(">>> ",
                                  history=FileHistory(os.path.join(self.config.model_dir, ".chat_history"))).strip()

    def test(self):
        start_test_time = time.time()

        assert self.config.n_responses >= 1

        if self.config.beam_width > 0:
            assert self.config.n_responses <= self.config.beam_width
        else:
            assert self.config.n_responses == 1

        self._pre_model_creation()

        infer_model = taware_helper.create_infer_model(taware_model.TopicAwareSeq2SeqModel, self.config)

        config_proto = models.model_helper.get_config_proto(self.config.log_device)

        ckpt = tf.train.latest_checkpoint(self.config.get_infer_model_dir())
        with tf.Session(graph=infer_model.graph, config=config_proto) as infer_sess:
            loaded_infer_model = model_helper.load_model(
                infer_model.model, ckpt, infer_sess, "infer")

            log.print_out("# Start decoding")
            log.print_out("  beam width: {}".format(self.config.beam_width))
            log.print_out("  length penalty: {}".format(self.config.length_penalty_weight))
            log.print_out("  sampling temperature: {}".format(self.config.sampling_temperature))
            log.print_out("  num responses per test instance: {}".format(self.config.n_responses))

            feed_dict = {
                infer_model.src_placeholder: self._load_data(self.config.test_data),
                infer_model.batch_size_placeholder: self.config.infer_batch_size,
            }

            infer_sess.run(infer_model.iterator.initializer, feed_dict=feed_dict)

            if self.config.sampling_temperature > 0:
                label = "%s_t%.1f" % (
                    fs.file_name(self.config.test_data), self.config.sampling_temperature)
            else:
                label = "%s_bw%d_lp%.1f" % (
                    fs.file_name(self.config.test_data), self.config.beam_width, self.config.length_penalty_weight)

            out_file = os.path.join(self.config.model_dir, "output_{}".format(label))

            eval_metric.decode_and_evaluate(
                "test",
                loaded_infer_model,
                infer_sess,
                out_file,
                ref_file=None,
                metrics=self.config.metrics,
                beam_width=self.config.beam_width,
                num_translations_per_input=self.config.n_responses)
        log.print_time("# Decoding done", start_test_time)

        eval_model = taware_helper.create_eval_model(taware_model.TopicAwareSeq2SeqModel, self.config)
        with tf.Session(
                config=models.model_helper.get_config_proto(self.config.log_device), graph=eval_model.graph) as eval_sess:
            loaded_eval_model = model_helper.load_model(
                eval_model.model, ckpt, eval_sess, "eval")

            log.print_out("# Compute Perplexity")

            feed_dict = {
                eval_model.eval_file_placeholder: self.config.test_data
            }

            eval_sess.run(eval_model.iterator.initializer, feed_dict=feed_dict)

            model_helper.compute_perplexity(loaded_eval_model, eval_sess, "test")

        log.print_time("# Test finished", start_test_time)
