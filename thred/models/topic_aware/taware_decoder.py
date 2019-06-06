import collections

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import _beam_search_step

from . import taware_layer


class ConservativeDecoderOutput(
    collections.namedtuple("ConservativeDecoderOutput", ("rnn_output", "prev_input", "context"))):
  pass


class ConservativeBasicDecoder(tf.contrib.seq2seq.BasicDecoder):
    def __init__(self, cell, helper, initial_state, output_layer):
        # if not isinstance(output_layer, taware_layer.JointDenseLayer):
        #     raise ValueError('Output layer must be of type: JointDenseLayer')

        self._current_context = None

        super(ConservativeBasicDecoder, self).__init__(cell, helper, initial_state, output_layer)

    def step(self, time, inputs, state, name=None):
        with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            if self._output_layer is not None:
                # My modification
                if isinstance(self._output_layer, taware_layer.JointDenseLayer):
                    if self._current_context is not None:
                        msg_attention, _ = tf.split(self._current_context, num_or_size_splits=2, axis=1)
                        cell_outputs = self._output_layer(cell_outputs, input=inputs, context=msg_attention)
                    else:
                        cell_outputs = self._output_layer(cell_outputs, input=inputs)
                else:
                    cell_outputs = self._output_layer(cell_outputs)

            sample_ids = self._helper.sample(
                time=time, outputs=cell_outputs, state=cell_state)
            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids)
            # My modification
            self._current_context = cell_state.attention

        outputs = tf.contrib.seq2seq.BasicDecoderOutput(cell_outputs, sample_ids)
        return (outputs, next_state, next_inputs, finished)


class ConservativeBeamSearchDecoder(tf.contrib.seq2seq.BeamSearchDecoder):
    def __init__(self,
                 cell,
                 embedding,
                 start_tokens,
                 end_token,
                 initial_state,
                 beam_width,
                 output_layer,
                 length_penalty_weight=0.0):
        # if not isinstance(output_layer, taware_layer.JointDenseLayer):
        #     raise ValueError('Output layer must be of type: JointDenseLayer')

        self._current_context = None

        super(ConservativeBeamSearchDecoder, self).__init__(cell,
                                                            embedding,
                                                            start_tokens,
                                                            end_token,
                                                            initial_state,
                                                            beam_width,
                                                            output_layer,
                                                            length_penalty_weight)

    def step(self, time, inputs, state, name=None):
        batch_size = self._batch_size
        beam_width = self._beam_width
        end_token = self._end_token
        length_penalty_weight = self._length_penalty_weight

        with ops.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
            cell_state = state.cell_state
            inputs = nest.map_structure(
                lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), inputs)
            cell_state = nest.map_structure(
                self._maybe_merge_batch_beams,
                cell_state, self._cell.state_size)
            cell_outputs, next_cell_state = self._cell(inputs, cell_state)
            cell_outputs = nest.map_structure(
                lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
            next_cell_state = nest.map_structure(
                self._maybe_split_batch_beams,
                next_cell_state, self._cell.state_size)

            if self._output_layer is not None:
                # My modification
                if isinstance(self._output_layer, taware_layer.JointDenseLayer):
                    reshaped_inputs = tf.reshape(inputs, [-1, beam_width, inputs.shape[-1]])
                    if self._current_context is not None:
                        msg_attention, _ = tf.split(self._current_context, num_or_size_splits=2, axis=1)
                        msg_attention = tf.reshape(msg_attention, [-1, beam_width, msg_attention.shape[-1]])
                        cell_outputs = self._output_layer(cell_outputs, input=reshaped_inputs, context=msg_attention)
                    else:
                        cell_outputs = self._output_layer(cell_outputs, input=reshaped_inputs)
                else:
                    cell_outputs = self._output_layer(cell_outputs)

            beam_search_output, beam_search_state = _beam_search_step(
                time=time,
                logits=cell_outputs,
                next_cell_state=next_cell_state,
                beam_state=state,
                batch_size=batch_size,
                beam_width=beam_width,
                end_token=end_token,
                length_penalty_weight=length_penalty_weight,
                coverage_penalty_weight=0.0)

            finished = beam_search_state.finished
            sample_ids = beam_search_output.predicted_ids
            next_inputs = control_flow_ops.cond(
                math_ops.reduce_all(finished), lambda: self._start_inputs,
                lambda: self._embedding_fn(sample_ids))

            # My modification
            self._current_context = cell_state.attention

        return (beam_search_output, beam_search_state, next_inputs, finished)
