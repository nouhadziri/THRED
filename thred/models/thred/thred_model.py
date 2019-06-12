from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import variable_scope

from .. import attention_helper
from ..base import AbstractModel
from ..topic_aware import taware_layer, taware_decoder
from thred.util import log, vocab, rnn_factory
from thred.util.device import DeviceManager, RoundRobin


class TopicAwareHierarchicalSeq2SeqModel(AbstractModel):
    """Topic-Aware Hierarchical Sequence-to-sequence model
    """

    def __init__(self,
                 mode,
                 num_turns,
                 iterator,
                 params,
                 rev_vocab_table=None,
                 scope=None,
                 log_trainables=True):

        log.print_out("# creating %s graph ..." % mode)
        self.dtype = tf.float32

        self.mode = mode
        self.num_turns = num_turns - 1

        self.device_manager = DeviceManager()
        self.round_robin = RoundRobin(self.device_manager)
        self.num_gpus = min(params.num_gpus, self.device_manager.num_available_gpus())
        log.print_out("# number of gpus %d" % self.num_gpus)

        self.iterator = iterator

        with tf.variable_scope(scope or 'thred_graph', dtype=self.dtype):
            self.init_embeddings(params.vocab_file, params.vocab_pkl, scope=scope)

            encoder_keep_prob, decoder_keep_prob = self.get_keep_probs(mode, params)
            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                context_keep_prob = 1.0 - params.context_dropout_rate
            else:
                context_keep_prob = 1.0

            with tf.variable_scope(scope or "build_network"):
                with tf.variable_scope("decoder/output_projection") as output_scope:
                    if params.boost_topic_gen_prob:
                        self.output_layer = taware_layer.JointDenseLayer(
                            params.vocab_size, params.topic_vocab_size,
                            scope=output_scope, name="output_projection")
                    else:
                        self.output_layer = layers_core.Dense(
                            params.vocab_size,
                            use_bias=False, name="output_projection")

            self.batch_size = tf.size(self.iterator.source_sequence_lengths[0])

            devices = self.round_robin.assign(2, base=self.num_gpus - 1)
            encoder_results, encoder_state = self.__build_encoder(params, encoder_keep_prob)
            context_outputs, context_state = self.__build_context(params, encoder_results, encoder_state,
                                                                  context_keep_prob, devices[0])

            self.global_step = tf.Variable(0, trainable=False)
            self.use_scheduled_sampling = False
            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                self.sampling_probability = tf.constant(params.scheduled_sampling_prob)
                self.sampling_probability = self._get_sampling_probability(params, self.global_step,
                                                                           self.sampling_probability)
                self.use_scheduled_sampling = params.scheduled_sampling_prob > 0
            elif mode == tf.contrib.learn.ModeKeys.EVAL:
                self.sampling_probability = tf.constant(0.0)

            logits, sample_ids, _ = self.__build_decoder(params, context_outputs, context_state,
                                                                           decoder_keep_prob, devices[1])

            if mode != tf.contrib.learn.ModeKeys.INFER:
                with tf.device(self.device_manager.tail_gpu()):
                    loss = self.__compute_loss(logits)
            else:
                loss, losses = None, None

            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                self.train_loss = loss
                self.word_count = sum(
                    [tf.reduce_sum(self.iterator.source_sequence_lengths[t]) for t in range(self.num_turns)]) + \
                                  tf.reduce_sum(self.iterator.target_sequence_length)
            elif mode == tf.contrib.learn.ModeKeys.EVAL:
                self.eval_loss = loss
            elif mode == tf.contrib.learn.ModeKeys.INFER:
                self.sample_words = rev_vocab_table.lookup(tf.to_int64(sample_ids))

            if mode != tf.contrib.learn.ModeKeys.INFER:
                ## Count the number of predicted words for compute ppl.
                self.predict_count = tf.reduce_sum(self.iterator.target_sequence_length)

            trainables = tf.trainable_variables()

            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                self.learning_rate = tf.constant(params.learning_rate)
                # decay
                self.learning_rate = self._get_learning_rate_decay(params, self.global_step, self.learning_rate)

                # Optimizer
                if params.optimizer.lower() == "sgd":
                    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                    tf.summary.scalar("lr", self.learning_rate)
                elif params.optimizer.lower() == "adam":
                    opt = tf.train.AdamOptimizer(self.learning_rate)
                    tf.summary.scalar("lr", self.learning_rate)
                else:
                    raise ValueError('Unknown optimizer: ' + params.optimizer)

                # Gradients
                gradients = tf.gradients(
                    self.train_loss,
                    trainables,
                    colocate_gradients_with_ops=True)

                clipped_grads, grad_norm = tf.clip_by_global_norm(gradients, params.max_gradient_norm)
                grad_norm_summary = [tf.summary.scalar("grad_norm", grad_norm)]
                grad_norm_summary.append(
                    tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_grads)))

                self.grad_norm = grad_norm

                self.update = opt.apply_gradients(
                    zip(clipped_grads, trainables), global_step=self.global_step)

                # Summary
                self.train_summary = tf.summary.merge([
                                                          tf.summary.scalar("lr", self.learning_rate),
                                                          tf.summary.scalar("train_loss", self.train_loss),
                                                      ] + grad_norm_summary)

            if mode == tf.contrib.learn.ModeKeys.INFER:
                self.infer_logits, self.sample_id = logits, sample_ids
                self.infer_summary = tf.no_op()

            # Saver
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

            # Print trainable variables
            if log_trainables:
                log.print_out("# Trainable variables")
                for trainable in trainables:
                    log.print_out("  %s, %s, %s" % (trainable.name, str(trainable.get_shape()),
                                                    trainable.op.device))

    def __build_encoder(self, params, keep_prob):
        encoder_cell = {}

        if params.encoder_type == "uni":
            encoder_cell['uni'] = rnn_factory.create_cell(params.cell_type, params.hidden_units,
                                                          use_residual=params.residual,
                                                          num_layers=params.num_layers,
                                                          input_keep_prob=keep_prob)
        elif params.encoder_type == "bi":
            num_bi_layers = int(params.num_layers / 2)
            encoder_cell['fw'] = rnn_factory.create_cell(params.cell_type, params.hidden_units,
                                                         use_residual=params.residual,
                                                         num_layers=num_bi_layers,
                                                         input_keep_prob=keep_prob)
            encoder_cell['bw'] = rnn_factory.create_cell(params.cell_type, params.hidden_units,
                                                         use_residual=params.residual,
                                                         num_layers=num_bi_layers,
                                                         input_keep_prob=keep_prob)
        else:
            raise ValueError("Unknown encoder type: '%s'" % params.encoder_type)

        encoding_devices = self.round_robin.assign(self.num_turns)

        encoder_results, next_initial_state = [], None
        for t in range(self.num_turns):
            with variable_scope.variable_scope("encoder") as scope:
                if t > 0:
                    scope.reuse_variables()

                with tf.device(encoding_devices[t]):
                    encoder_embedded_inputs = tf.nn.embedding_lookup(params=self.embeddings,
                                                                     ids=self.iterator.sources[t])

                    if params.encoder_type == "bi":
                        encoder_outputs, states = tf.nn.bidirectional_dynamic_rnn(
                            encoder_cell['fw'],
                            encoder_cell['bw'],
                            inputs=encoder_embedded_inputs,
                            dtype=self.dtype,
                            sequence_length=self.iterator.source_sequence_lengths[t],
                            swap_memory=True)

                        fw_state, bw_state = states
                        num_bi_layers = int(params.num_layers / 2)
                        if t == 0:
                            if params.context_type == "uni":
                                next_initial_state = self._merge_bidirectional_states(num_bi_layers, fw_state, bw_state)
                            else:
                                if num_bi_layers > 1:
                                    initial_state_fw, initial_state_bw = [], []
                                    for layer_id in range(num_bi_layers):
                                        initial_state_fw.append(fw_state[layer_id])
                                        initial_state_bw.append(bw_state[layer_id])

                                    next_initial_state = (tuple(initial_state_fw), tuple(initial_state_bw))
                                else:
                                    next_initial_state = (fw_state, bw_state)

                        if num_bi_layers > 1:
                            next_input = tf.concat([fw_state[-1], bw_state[-1]], axis=1)
                        else:
                            next_input = tf.concat([fw_state, bw_state], axis=1)
                    else:
                        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                            encoder_cell['uni'],
                            inputs=encoder_embedded_inputs,
                            sequence_length=self.iterator.source_sequence_lengths[t],
                            dtype=self.dtype,
                            swap_memory=True,
                            scope=scope)

                        if t == 0:
                            if params.context_type == "uni":
                                next_initial_state = encoder_state
                            else:
                                num_bi_layers = int(params.num_layers / 2)
                                initial_state_fw, initial_state_bw = [], []
                                for layer_id in range(num_bi_layers):
                                    initial_state_fw.append(encoder_state[2 * layer_id])
                                    initial_state_bw.append(encoder_state[2 * layer_id + 1])
                                next_initial_state = (tuple(initial_state_fw), tuple(initial_state_bw))

                        if params.num_layers > 1:
                            next_input = encoder_state[-1]
                        else:
                            next_input = encoder_state

                    # msg_attn_mechanism = attention_helper.create_attention_mechanism(
                    #     params.attention_type,
                    #     params.hidden_units,
                    #     encoder_outputs,
                    #     self.iterator.source_sequence_lengths[t])

                    encoder_results.append((encoder_outputs, next_input))

        return encoder_results, next_initial_state

    def _merge_bidirectional_states(self, num_bi_layers, fw_state, bw_state):
        if num_bi_layers > 1:
            merged_state = []
            for layer_id in range(num_bi_layers):
                merged_state.append(fw_state[layer_id])
                merged_state.append(bw_state[layer_id])
            merged_state = tuple(merged_state)
        else:
            merged_state = (fw_state, bw_state)
        return merged_state

    def __build_context(self, params, encoder_results, initial_state, keep_prob, device):
        with variable_scope.variable_scope("context") as scope:
            with tf.device(device):
                context_seq_length = tf.fill([self.batch_size], self.num_turns)
                context_inputs = tf.stack([state for _, state in encoder_results], axis=0)

                # message_attention = attention_helper.create_attention_mechanism(params.attention_type,
                #                                                                 params.hidden_units,
                #                                                                 context_inputs)

                if params.context_type == "uni":
                    cell = rnn_factory.create_cell(params.cell_type, params.hidden_units,
                                                   use_residual=params.residual, num_layers=params.num_layers,
                                                   input_keep_prob=keep_prob)

                    # cell = tf.contrib.seq2seq.AttentionWrapper(
                    #     cell,
                    #     msg_attn_mechanism,
                    #     attention_layer_size=params.hidden_units,
                    #     alignment_history=False,
                    #     output_attention=True,
                    #     name="message_attention")
                    context_outputs, context_state = tf.nn.dynamic_rnn(cell,
                                                                       initial_state=initial_state,
                                                                       inputs=context_inputs,
                                                                       sequence_length=context_seq_length,
                                                                       time_major=True,
                                                                       dtype=self.dtype,
                                                                       swap_memory=True)

                    return context_outputs, context_state
                elif params.context_type == "bi":
                    num_bi_layers = int(params.num_layers / 2)
                    fw_cell = rnn_factory.create_cell(params.cell_type, params.hidden_units, num_bi_layers,
                                                      use_residual=params.residual,
                                                      input_keep_prob=keep_prob,
                                                      devices=self.round_robin.assign(num_bi_layers))
                    bw_cell = rnn_factory.create_cell(params.cell_type, params.hidden_units, num_bi_layers,
                                                      use_residual=params.residual,
                                                      input_keep_prob=keep_prob,
                                                      devices=self.round_robin.assign(num_bi_layers,
                                                                                      self.device_manager.num_available_gpus() - 1))

                    context_outputs, context_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                                                     context_inputs,
                                                                                     initial_state_fw=initial_state[0],
                                                                                     initial_state_bw=initial_state[1],
                                                                                     sequence_length=context_seq_length,
                                                                                     time_major=True,
                                                                                     dtype=scope.dtype,
                                                                                     swap_memory=True)

                    fw_state, bw_state = context_state
                    fw_output, bw_output = context_outputs
                    context_outputs = tf.concat([fw_output, bw_output], axis=-1)
                    return context_outputs, self._merge_bidirectional_states(num_bi_layers, fw_state, bw_state)
                else:
                    raise ValueError("Unknown context type: %s" % params.context_type)

    def __build_decoder_cell(self, params, context_outputs, context_state, input_keep_prob):
        cell = rnn_factory.create_cell(params.cell_type, params.hidden_units,
                                       use_residual=params.residual, num_layers=params.num_layers,
                                       input_keep_prob=input_keep_prob,
                                       devices=self.round_robin.assign(params.num_layers))

        topical_embeddings = tf.nn.embedding_lookup(self.embeddings, self.iterator.topic)

        max_topic_length = tf.reduce_max(self.iterator.topic_sequence_length)

        expanded_context_state = tf.tile(tf.expand_dims(context_state[-1] if params.num_layers > 1 else context_state, axis=1),
                                         [1, max_topic_length, 1])
        topical_embeddings = tf.concat([expanded_context_state, topical_embeddings], axis=2)

        context_sequence_length = tf.fill([self.batch_size], self.num_turns)
        batch_majored_context_outputs = tf.transpose(context_outputs, [1, 0, 2])

        if self.mode == tf.contrib.learn.ModeKeys.INFER and params.beam_width > 0:
            batch_size = self.batch_size * params.beam_width

            decoder_initial_state = tf.contrib.seq2seq.tile_batch(context_state, multiplier=params.beam_width)

            memory = tf.contrib.seq2seq.tile_batch(batch_majored_context_outputs, multiplier=params.beam_width)
            topical_embeddings = tf.contrib.seq2seq.tile_batch(topical_embeddings, multiplier=params.beam_width)
            context_sequence_length = tf.contrib.seq2seq.tile_batch(
                context_sequence_length, multiplier=params.beam_width)
            topic_sequence_length = tf.contrib.seq2seq.tile_batch(
                self.iterator.topic_sequence_length, multiplier=params.beam_width)
        else:
            batch_size = self.batch_size
            decoder_initial_state = context_state
            memory = batch_majored_context_outputs
            topic_sequence_length = self.iterator.topic_sequence_length

        context_attention = attention_helper.create_attention_mechanism(params.attention_type,
                                                                        params.hidden_units,
                                                                        memory, context_sequence_length)

        topical_attention = attention_helper.create_attention_mechanism(params.attention_type,
                                                                        params.hidden_units,
                                                                        topical_embeddings, topic_sequence_length)

        alignment_history = self.mode == tf.contrib.learn.ModeKeys.INFER and params.beam_width == 0
        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism=(context_attention, topical_attention),
            attention_layer_size=(params.hidden_units, params.hidden_units),
            alignment_history=alignment_history,
            output_attention=True,
            name="joint_attention")

        decoder_initial_state = cell.zero_state(batch_size, self.dtype).clone(cell_state=decoder_initial_state)

        return cell, decoder_initial_state

    def __build_decoder(self, params, context_outputs, context_state,
                        keep_prob, device):
        iterator = self.iterator
        with variable_scope.variable_scope("decoder") as scope:
            with tf.device(device):
                cell, initial_state = self.__build_decoder_cell(params, context_outputs, context_state, keep_prob)

                if self.mode != tf.contrib.learn.ModeKeys.INFER:
                    # decoder_emp_inp: [max_time, batch_size, num_units]
                    decoder_emb_inp = tf.nn.embedding_lookup(self.embeddings, iterator.target_input)

                    # Helper
                    if self.use_scheduled_sampling:
                        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                            decoder_emb_inp, iterator.target_sequence_length, self.embeddings,
                            self.sampling_probability)
                    else:
                        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, iterator.target_sequence_length)

                    # Decoder
                    my_decoder = taware_decoder.ConservativeBasicDecoder(
                        cell,
                        helper,
                        initial_state,
                        self.output_layer)

                    # Dynamic decoding
                    outputs, final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder,
                                                                                        swap_memory=True,
                                                                                        scope=scope)

                    sample_ids = outputs.sample_id
                    logits = outputs.rnn_output

                    # Note: there's a subtle difference here between train and inference.
                    # We could have set output_layer when create my_decoder
                    #   and shared more code between train and inference.
                    # We chose to apply the output_layer to all timesteps for speed:
                    #   10% improvements for small models & 20% for larger ones.
                    # If memory is a concern, we should apply output_layer per timestep.

                ### Inference
                else:
                    beam_width = params.beam_width
                    start_tokens = tf.fill([self.batch_size], vocab.SOS_ID)
                    end_token = vocab.EOS_ID

                    maximum_iterations = self._get_decoder_max_iterations(params)

                    if beam_width > 0:
                        # initial_state = tf.contrib.seq2seq.tile_batch(context_outputs[-1],
                        #                                               multiplier=params.beam_width)

                        my_decoder = taware_decoder.ConservativeBeamSearchDecoder(
                            cell, self.embeddings, start_tokens, end_token,
                            initial_state=initial_state,
                            beam_width=beam_width,
                            output_layer=self.output_layer,
                            length_penalty_weight=params.length_penalty_weight)
                    else:
                        # Helper
                        if params.sampling_temperature > 0.0:
                            helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                                self.embeddings, start_tokens, end_token,
                                softmax_temperature=params.sampling_temperature,
                                seed=None)
                        else:
                            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings, start_tokens, end_token)

                        # Decoder
                        my_decoder = taware_decoder.ConservativeBasicDecoder(
                            cell,
                            helper,
                            initial_state,
                            output_layer=self.output_layer  # applied per timestep
                        )

                    # Dynamic decoding
                    outputs, final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder,
                                                                                        maximum_iterations=maximum_iterations,
                                                                                        swap_memory=True,
                                                                                        scope=scope)

                    if beam_width > 0:
                        logits = tf.no_op()
                        sample_ids = outputs.predicted_ids
                    else:
                        logits = outputs.rnn_output
                        sample_ids = outputs.sample_id

        return logits, sample_ids, final_decoder_state

    def _get_decoder_max_iterations(self, params):
        max_encoder_length = None
        for t in range(self.num_turns):
            if max_encoder_length is None:
                max_encoder_length = tf.reduce_max(self.iterator.source_sequence_lengths[t])
            else:
                max_encoder_length = tf.maximum(max_encoder_length,
                                                tf.reduce_max(self.iterator.source_sequence_lengths[t]))
        return tf.to_int32(
            tf.round(tf.to_float(max_encoder_length) * params.decoding_length_factor))

    def __compute_loss(self, logits):
        iterator = self.iterator
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=iterator.target_output, logits=logits)

        max_time = iterator.target_output.shape[1].value or tf.shape(iterator.target_output)[1]
        target_weights = tf.sequence_mask(iterator.target_sequence_length, max_time, dtype=self.dtype)

        loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.batch_size)
        return loss

    def train(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN

        return sess.run([self.update,
                         self.train_loss,
                         self.predict_count,
                         self.train_summary,
                         self.global_step,
                         self.word_count,
                         self.batch_size,
                         self.grad_norm,
                         self.learning_rate])

    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL

        return sess.run(
            [self.eval_loss, self.predict_count, self.batch_size])

    def infer(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER

        return sess.run(
            [self.infer_logits, self.infer_summary, self.sample_id, self.sample_words])

    def decode(self, sess):
        _, infer_summary, _, sample_words = self.infer(sess)
        if sample_words.ndim == 3:  # beam search output in [batch_size,
            # time, beam_width] shape.
            sample_words = sample_words.transpose([2, 0, 1])

        return sample_words, infer_summary
