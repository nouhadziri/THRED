
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


def create_cell(unit_type, hidden_units, num_layers, use_residual=False, input_keep_prob=1.0, output_keep_prob=1.0, devices=None):
    if unit_type == 'lstm':
        def _new_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
    elif unit_type == 'gru':
        def _new_cell():
            return tf.contrib.rnn.GRUCell(hidden_units)
    else:
        raise ValueError('cell_type must be either lstm or gru')

    def _new_cell_wrapper(residual_connection=False, device_id=None):
        c = _new_cell()

        if input_keep_prob < 1.0 or output_keep_prob < 1.0:
            c = rnn.DropoutWrapper(c, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)

        if residual_connection:
            c = rnn.ResidualWrapper(c)

        if device_id:
            c = rnn.DeviceWrapper(c, device_id)

        return c

    if num_layers > 1:
        cells = []

        for i in range(num_layers):
            is_residual = True if use_residual and i > 0 else False
            cells.append(_new_cell_wrapper(is_residual, devices[i] if devices else None))

        return tf.contrib.rnn.MultiRNNCell(cells)
    else:
        return _new_cell_wrapper(device_id=devices[0] if devices else None)
