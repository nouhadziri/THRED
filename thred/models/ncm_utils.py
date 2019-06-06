from ..util import vocab


def get_translation(ncm_outputs, sent_id):
    """Given batch decoding outputs, select a sentence and turn to text."""
    # Select a sentence
    output = ncm_outputs[sent_id, :].tolist() if len(ncm_outputs.shape) > 1 else ncm_outputs.tolist()

    eos = vocab.EOS.encode("utf-8")

    # If there is an eos symbol in outputs, cut them at that point.
    if eos in output:
        output = output[:output.index(eos)]

    return b" ".join(output)
