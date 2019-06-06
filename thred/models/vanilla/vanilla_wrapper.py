import os
import random
import time

import math
import tensorflow as tf
from tqdm import trange

from thred.util import fs, log
from thred.util.misc import Stopwatch
from . import vanilla_helper
from .. import ncm_utils, model_helper
from ..base import NMTEncoderDecoderWrapper


class VanillaNMTEncoderDecoder(NMTEncoderDecoderWrapper):
    def __init__(self, config) -> None:
        super(VanillaNMTEncoderDecoder, self).__init__(config)

    def _get_checkpoint_name(self):
        return 'vanilla_nmt'

    def _get_model_helper(self):
        return vanilla_helper

    def train(self, target_session="", scope=None):
        out_dir = self.config.model_dir
        model_dir = out_dir

        steps_per_stats = self.config.steps_per_stats
        steps_per_eval = self.config.steps_per_eval

        self._pre_model_creation()

        train_model = vanilla_helper.create_train_model(self.config, scope)
        eval_model = vanilla_helper.create_eval_model(self.config, scope)
        infer_model = vanilla_helper.create_infer_model(self.config, scope)

        # Preload data for sample decoding.
        eval_data = self._load_data(self.config.dev_data)
        self.config.dev_size = math.ceil(len(eval_data) / self.config.batch_size)

        summary_name = "train_log"

        # Log and output files
        log_file = os.path.join(out_dir, "log_%d" % time.time())
        log_f = tf.gfile.GFile(log_file, mode="a")
        log.print_out("# log_file=%s" % log_file, log_f)

        avg_step_time = 0.0

        # TensorFlow model
        config_proto = model_helper.get_config_proto(self.config.log_device)

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

        self.config.save()

        # Initialize all of the iterators
        skip_count = self.config.batch_size * self.config.epoch_step
        lr = loaded_train_model.learning_rate.eval(session=train_sess)
        log.print_out(
            "# Starting step {}/{} (skipping {} elements), epoch {}/{}, lr {:f}, {}".format(
                global_step, self.config.num_train_steps, skip_count,
                self.config.epoch, self.config.num_train_epochs, lr, time.ctime()),
            log_f)

        train_sess.run(
            train_model.iterator.initializer,
            feed_dict={train_model.skip_count_placeholder: skip_count})

        pbar = trange(self.config.num_train_steps, initial=global_step)
        pbar.set_postfix(lr=lr, wps='0K', ppl='inf', gN='inf', best_dev_ppl=self.config.best_dev_ppl)
        pbar.set_description("Ep {}/{}".format(self.config.epoch, self.config.num_train_epochs))

        while self.config.epoch < self.config.num_train_epochs and patience > 0:
            ### Run a step ###
            start_time = time.time()
            try:
                step_result = loaded_train_model.train(train_sess)
                self.config.epoch_step += 1
            except tf.errors.OutOfRangeError:
                # Finished going through the training dataset.  Go to next epoch.
                sw = Stopwatch()
                self.run_sample_decode(infer_model, infer_sess,
                                       model_dir, summary_writer, eval_data)

                log.print_out(
                    "## Done epoch {} in {} steps. step {} @ eval time: {}s".format(
                        self.config.epoch, self.config.epoch_step, global_step, sw.elapsed()))

                self.config.epoch += 1
                self.config.epoch_step = 0
                self.config.save()
                pbar.set_description("Ep {}/{}".format(self.config.epoch, self.config.num_train_epochs))

                # dev_scores, test_scores, _ = self.run_external_eval(infer_model, infer_sess, model_dir, summary_writer)
                train_sess.run(
                    train_model.iterator.initializer,
                    feed_dict={train_model.skip_count_placeholder: 0})
                continue

            # Write step summary and accumulate statistics
            global_step = self.update_stats(stats, summary_writer, start_time, step_result)

            # Once in a while, we print statistics.
            if global_step - last_stats_step >= steps_per_stats:
                train_ppl, speed, is_overflow = self.check_stats(stats, global_step, steps_per_stats, log_f, pbar)
                pbar.update(global_step - last_stats_step)
                last_stats_step = global_step

                if is_overflow:
                    break

                # Reset statistics
                stats = self.init_stats()

            if global_step - last_eval_step >= steps_per_eval:
                last_eval_step = global_step

                log.print_out("# Save eval, global step {}".format(global_step))
                log.add_summary(summary_writer, global_step, "train_ppl", train_ppl)

                # Save checkpoint
                loaded_train_model.saver.save(
                    train_sess,
                    os.path.join(out_dir, "vanilla.ckpt"),
                    global_step=global_step)

                # Evaluate on dev
                self.run_sample_decode(infer_model, infer_sess, model_dir, summary_writer, eval_data)
                dev_ppl = self.run_internal_eval(eval_model, eval_sess, model_dir, summary_writer)

                if dev_ppl < self.config.best_dev_ppl:
                    self.config.best_dev_ppl = dev_ppl
                    patience = self.config.patience
                    log.print_out('    **** Best model so far @Ep {} @step {} (global {}) dev_ppl {:.3f}'.format(
                        self.config.epoch,
                        self.config.epoch_step,
                        global_step,
                        dev_ppl))
                elif dev_ppl > self.config.degrade_threshold * self.config.best_dev_ppl:
                    patience -= 1
                    log.print_out(
                        '    PPL got worse @Ep {} @step {} (global {}) patience {} '
                        'dev_ppl {:.3f} best_dev_ppl {:.3f}'.format(
                            self.config.epoch,
                            self.config.epoch_step,
                            global_step,
                            patience,
                            dev_ppl,
                            self.config.best_dev_ppl))

                # Save config parameters
                self.config.save()

        pbar.close()
        # Done training
        loaded_train_model.saver.save(
            train_sess,
            os.path.join(out_dir, "vanilla.ckpt"),
            global_step=global_step)

        dev_scores, test_scores, dev_ppl, test_ppl = None, None, None, None

        log.print_out(
            "# Final, step {} ep {}/{} lr {:f} "
            "step-time {:.2f} wps {:.2f}K train_ppl {:.3f} best_dev_ppl {:.3f}, {}".format(
                global_step, self.config.epoch, self.config.epoch_step,
                loaded_train_model.learning_rate.eval(session=train_sess),
                avg_step_time, speed, train_ppl, self.config.best_dev_ppl, time.ctime()),
            log_f)
        log.print_time("# Done training!", start_train_time)

        if self.config.eval_best_model:
            log.print_out("Evaluating the best model begins...")
            test_ppl = self.run_infer_best_model(infer_model, eval_model,
                                                 infer_sess, eval_sess,
                                                 self.config.best_dev_ppl_dir,
                                                 fs.file_name(self.config.test_data) + '_final',
                                                 summary_writer)

            log.print_out(
                "# test_ppl {:.3f} w. beam_width: {} | length_penalty: {:.1f} | sampling_temperature: {:.1f}".format
                (test_ppl, self.config.beam_width, self.config.length_penalty_weight, self.config.sampling_temperature),
                log_f)

        summary_writer.close()

        return (dev_scores, test_scores, dev_ppl, test_ppl, global_step)

    def _sample_decode(self,
                       model, global_step, sess,
                       iterator_src_placeholder, iterator_batch_size_placeholder,
                       eval_data, summary_writer):
        """Pick a sentence and decode."""
        decode_id = random.randint(0, len(eval_data) - 1)
        log.print_out("  # {}".format(decode_id))

        iterator_feed_dict = {
            iterator_src_placeholder: [eval_data[decode_id]],
            iterator_batch_size_placeholder: 1,
        }
        sess.run(model.iterator.initializer, feed_dict=iterator_feed_dict)

        ncm_outputs, attention_summary = model.decode(sess)

        if self.config.beam_width > 0:
            # get the top translation.
            ncm_outputs = ncm_outputs[0]

        translation = ncm_utils.get_translation(ncm_outputs, sent_id=0)
        log.print_out("    sources:")

        utterances = eval_data[decode_id].split("\t")
        sources, target = utterances[:-1], utterances[-1]

        for t, src in enumerate(sources):
            log.print_out("      @{} {}".format(t + 1, src))
        log.print_out("    resp: {}".format(target))
        log.print_out(b"    generated: " + translation)

        # Summary
        if attention_summary is not None:
            summary_writer.add_summary(attention_summary, global_step)

    def interactive(self, scope=None):
        self.config.num_turns = 3
        super().interactive(scope)

    def test(self):
        start_test_time = time.time()

        assert self.config.n_responses >= 1

        if self.config.beam_width > 0:
            assert self.config.n_responses <= self.config.beam_width
        else:
            assert self.config.n_responses == 1

        self._pre_model_creation()

        infer_model = vanilla_helper.create_infer_model(self.config)

        config_proto = model_helper.get_config_proto(self.config.log_device)

        ckpt = tf.train.latest_checkpoint(self.config.get_infer_model_dir())
        with tf.Session(graph=infer_model.graph, config=config_proto) as infer_sess:
            loaded_infer_model = model_helper.load_model(
                infer_model.model, ckpt, infer_sess, "infer")

            log.print_out("# Start decoding")
            log.print_out("  beam width: {}".format(self.config.beam_width))
            log.print_out("  length penalty: {}".format(self.config.length_penalty_weight))
            log.print_out("  sampling temperature: {}".format(self.config.sampling_temperature))
            log.print_out("  num responses per tests instance: {}".format(self.config.n_responses))

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

            self._decode_and_evaluate(loaded_infer_model, infer_sess, feed_dict,
                                      label=label,
                                      num_responses_per_input=self.config.n_responses)
        log.print_time("# Decoding done", start_test_time)

        eval_model = vanilla_helper.create_eval_model(self.config)
        with tf.Session(
                config=model_helper.get_config_proto(self.config.log_device),
                graph=eval_model.graph) as eval_sess:
            loaded_eval_model = model_helper.load_model(
                eval_model.model, ckpt, eval_sess, "eval")

            log.print_out("# Compute Perplexity")

            feed_dict = {
                eval_model.eval_file_placeholder: self.config.test_data
            }

            eval_sess.run(eval_model.iterator.initializer, feed_dict=feed_dict)

            model_helper.compute_perplexity(loaded_eval_model, eval_sess, "test")

        log.print_time("# Test finished", start_test_time)
