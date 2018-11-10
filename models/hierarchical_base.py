import codecs
import os
import time

import numpy as np
import tensorflow as tf

from models import model_helper
from models import ncm_utils
from models.base import NMTEncoderDecoderWrapper
from util import log, fs, vocab
from util.misc import Stopwatch


class BaseHierarchicalEncoderDecoder(NMTEncoderDecoderWrapper):
    def __init__(self, config):
        super(BaseHierarchicalEncoderDecoder, self).__init__(config)

    def pretrain(self, pretrain_sess, pretrain_model, log_f):
        pretrain_sess.run(tf.global_variables_initializer())
        pretrain_sess.run(tf.tables_initializer())

        global_step = pretrain_model.model.global_step.eval(session=pretrain_sess)

        num_pretrain_steps = self.config.num_pretrain_steps
        epoch, epoch_step = 0, 0

        summary_name = "pretrain_log"

        summary_writer = tf.summary.FileWriter(
            os.path.join(self.config.model_dir, summary_name), pretrain_model.graph)

        last_stats_step = global_step

        # This is the training loop.
        stats = self.init_stats()
        speed, train_ppl = 0.0, 0.0

        log.print_out("%% Pretraining starts for %d steps -> step %d, lr %g, %s" %
                      (self.config.num_pretrain_steps, global_step,
                       pretrain_model.model.learning_rate.eval(session=pretrain_sess), time.ctime()), log_f)

        pretrain_sess.run(
            pretrain_model.iterator.initializer,
            feed_dict={pretrain_model.skip_count_placeholder: 0})

        # pretrain_sw = Stopwatch()
        while global_step < num_pretrain_steps:
            ### Run a step ###
            start_time = time.time()

            try:
                step_result = pretrain_model.model.train(pretrain_sess)
                epoch_step += 1
            except tf.errors.OutOfRangeError:
                epoch_step = 0
                epoch += 1

                log.print_out(
                    "%% Pretraining: finished epoch %d, step %d." % (epoch, global_step))
                pretrain_sess.run(
                    pretrain_model.iterator.initializer,
                    feed_dict={pretrain_model.skip_count_placeholder: 0})
                continue

            # Write step summary and accumulate statistics
            global_step = self.update_stats(stats, summary_writer, start_time, step_result)

            # Once in a while, we print statistics.
            if global_step - last_stats_step >= self.config.steps_per_stats:
                last_stats_step = global_step
                train_ppl, speed, is_overflow = self.check_stats(stats, global_step, self.config.steps_per_stats, log_f)
                if is_overflow:
                    break

                # Reset statistics
                stats = self.init_stats()

        log.print_out("%% Pretraining finished at step %d" % global_step)
        pretrain_model.model.saver.save(
            pretrain_sess,
            self.config.checkpoint_file,
            global_step=global_step)

    def train(self, target_session="", scope=None):
        assert self.config.num_turns >= 2
        if self.config.is_pretrain_enabled():
            assert self.config.num_pretrain_turns >= 2
            assert self.config.num_turns >= self.config.num_pretrain_turns

        out_dir = self.config.model_dir

        steps_per_stats = self.config.steps_per_stats
        steps_per_eval = 20 * steps_per_stats

        _helper = self._get_model_helper()

        self._pre_model_creation()

        train_model = _helper.create_train_model(self.config, scope)
        eval_model = _helper.create_eval_model(self.config, scope)
        infer_model = _helper.create_infer_model(self.config, scope)

        self._post_model_creation(train_model, eval_model, infer_model)

        # Preload data for sample decoding.
        dev_file = self.config.dev_data
        eval_data = self._load_data(dev_file, include_target=True)

        summary_name = "train_log"

        # Log and output files
        log_file = os.path.join(out_dir, "log_%d" % time.time())
        log_f = tf.gfile.GFile(log_file, mode="a")
        log.print_out("# log_file=%s" % log_file, log_f)

        self.config.save()
        log.print_out("# Configs saved")

        avg_step_time = 0.0

        # TensorFlow model
        config_proto = model_helper.get_config_proto(self.config.log_device)

        train_sess = tf.Session(
            target=target_session, config=config_proto, graph=train_model.graph)
        eval_sess = tf.Session(
            target=target_session, config=config_proto, graph=eval_model.graph)
        infer_sess = tf.Session(
            target=target_session, config=config_proto, graph=infer_model.graph)

        # Pretraining
        num_pretrain_steps = 0
        if self.config.is_pretrain_enabled():
            num_pretrain_steps = self.config.num_pretrain_steps

            pretrain_model = _helper.create_pretrain_model(self.config, scope)

            with tf.Session(
                    target=target_session, config=config_proto, graph=pretrain_model.graph) as pretrain_sess:
                self.pretrain(pretrain_sess, pretrain_model, log_f)

        with train_model.graph.as_default():
            loaded_train_model, global_step = model_helper.create_or_load_model(
                train_model.model, self.config.model_dir, train_sess, "train")

        # Summary writer
        summary_writer = tf.summary.FileWriter(
            os.path.join(out_dir, summary_name), train_model.graph)

        last_stats_step = global_step
        last_eval_step = global_step
        patience = self.config.patience

        stats = self.init_stats()
        speed, train_ppl = 0.0, 0.0
        start_train_time = time.time()

        log.print_out(
            "# Start step %d, epoch %d, lr %g, %s" %
            (global_step, self.config.epoch, loaded_train_model.learning_rate.eval(session=train_sess),
             time.ctime()),
            log_f)

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
                self.run_sample_decode(infer_model, infer_sess,
                                       self.config.model_dir, summary_writer, eval_data)
                # if self.config.enable_epoch_evals:
                #     dev_ppl, test_ppl = self.run_full_eval(infer_model, eval_model,
                #                                            infer_sess, eval_sess,
                #                                            out_dir,
                #                                            fs.file_name(self.config.test_data) + '_' + global_step,
                #                                            summary_writer)
                #     log.print_out(
                #         "%% done epoch %d #%d  step %d - dev_ppl: %.2f test_ppl: %.2f @ eval time: %ds" %
                #         (self.config.epoch, self.config.epoch_step, global_step, dev_ppl, test_ppl, sw.elapsed()))
                # else:
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
                loaded_train_model.saver.save(train_sess,
                                              self.config.checkpoint_file,
                                              global_step=global_step)

                # Evaluate on dev
                self.run_sample_decode(infer_model, infer_sess, out_dir, summary_writer, eval_data)
                dev_ppl, _ = self.run_internal_eval(eval_model, eval_sess, out_dir, summary_writer,
                                                    use_test_set=False)
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
                            patience,
                            self.config.best_dev_ppl))

                # Save config parameters
                self.config.save()

        # Done training
        loaded_train_model.saver.save(
            train_sess,
            self.config.checkpoint_file,
            global_step=global_step)

        if self.config.enable_final_eval:
            dev_ppl, test_ppl = self.run_full_eval(infer_model, eval_model,
                                                   infer_sess, eval_sess,
                                                   out_dir,
                                                   fs.file_name(self.config.test_data) + '_final',
                                                   summary_writer)

            log.print_out(
                "# Final, step %d ep %d/%d lr %g "
                "step-time %.2f wps %.2fK train_ppl %.2f, dev_ppl %.2f, test_ppl %.2f, %s" %
                (global_step, self.config.epoch, self.config.epoch_step,
                 loaded_train_model.learning_rate.eval(session=train_sess),
                 avg_step_time, speed, train_ppl, dev_ppl, test_ppl, time.ctime()),
                log_f)
        else:
            log.print_out(
                "# Final, step %d ep %d/%d lr %g "
                "step-time %.2f wps %.2fK train_ppl %.2f best_dev_ppl %.2f, %s" %
                (global_step, self.config.epoch, self.config.epoch_step,
                 loaded_train_model.learning_rate.eval(session=train_sess),
                 avg_step_time, speed, train_ppl, self.config.best_dev_ppl, time.ctime()),
                log_f)

        log.print_time("# Done training!", start_train_time)

        summary_writer.close()

        eval_sess.close()
        infer_sess.close()
        train_sess.close()

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
        #
        # return (None, None, dev_ppl, test_ppl, global_step)

    def _load_data(self, input_file, include_target=False):
        """Load inference data."""
        inference_data = []
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(input_file, mode="rb")) as f:
            for line in f:
                utterances = line.split('\t')
                sources = '\t'.join(src.strip() for src in utterances[:-1])

                if include_target:
                    target = utterances[-1].strip()
                    inference_data.append((sources, target))
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
            for t, src in enumerate(eval_data[decode_id][0].split('\t')):
                log.print_out("      @%d %s" % (t + 1, src))
            log.print_out("    resp: %s" % eval_data[decode_id][1])
            log.print_out(b"    generated: " + translation)

        # Summary
        if infer_summary is not None:
            summary_writer.add_summary(infer_summary, global_step)

    def interactive(self, scope=None):
        import platform
        from prompt_toolkit import prompt
        from prompt_toolkit.history import FileHistory
        from topic_model.lda import TopicInferer, DEFAULT_SEPARATOR
        from util.nlp import NLPToolkit

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
            latest_ckpt = tf.train.latest_checkpoint(self.config.model_dir)
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
                    topic_words = topic_inferer.from_collection([feedable_context],
                                                                dialogue_as_doc=True,
                                                                words_per_topic=self.config.topic_words_per_utterance)
                    iterator_feed_dict = {
                        infer_model.src_placeholder:
                            [feedable_context + DEFAULT_SEPARATOR + " ".join(topic_words)],
                        infer_model.batch_size_placeholder: 1,
                    }

                sess.run(infer_model.iterator.initializer, feed_dict=iterator_feed_dict)
                output, infer_summary = loaded_infer_model.decode(sess)
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

    def test(self):
        start_test_time = time.time()

        assert self.config.n_responses >= 1

        if self.config.beam_width > 0:
            assert self.config.n_responses <= self.config.beam_width
        else:
            assert self.config.n_responses == 1

        self._pre_model_creation()

        infer_model = self._get_model_helper().create_infer_model(self.config)

        latest_ckpt = tf.train.latest_checkpoint(self.config.get_infer_model_dir())
        with tf.Session(
                config=model_helper.get_config_proto(self.config.log_device),
                graph=infer_model.graph) as infer_sess:
            loaded_infer_model = model_helper.load_model(
                infer_model.model, latest_ckpt, infer_sess, "infer")

            log.print_out("# Start decoding")
            log.print_out("  beam width: {}".format(self.config.beam_width))
            log.print_out("  length penalty: {}".format(self.config.length_penalty_weight))
            log.print_out("  sampling temperature: {}".format(self.config.sampling_temperature))
            log.print_out("  num responses per test instance: {}".format(self.config.n_responses))

            feed_dict = {
                infer_model.src_placeholder: self._load_data(self.config.test_data),
                infer_model.batch_size_placeholder: self.config.infer_batch_size,
            }

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

        eval_model = self._get_model_helper().create_eval_model(self.config)
        with tf.Session(
                config=model_helper.get_config_proto(self.config.log_device),
                graph=eval_model.graph) as eval_sess:
            loaded_eval_model = model_helper.load_model(
                eval_model.model, latest_ckpt, eval_sess, "eval")

            log.print_out("# Compute Perplexity")

            dev_eval_iterator_feed_dict = {
                eval_model.eval_file_placeholder: self.config.test_data
            }

            eval_sess.run(eval_model.iterator.initializer, feed_dict=dev_eval_iterator_feed_dict)
            model_helper.compute_perplexity(loaded_eval_model, eval_sess, "test")

        log.print_time("# Test finished", start_test_time)
