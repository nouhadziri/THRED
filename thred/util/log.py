import sys
import time

import tensorflow as tf


def print_time(s, start_time):
    """Take a start time, print elapsed duration, and return a new time."""

    print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
    # sys.stdout.flush()
    return time.time()


def print_out(s, f=None, new_line=True, skip_stdout=False):
    """Similar to print but with support to flush and output to a file."""

    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")

    # stdout
    if not skip_stdout:
        out_s = s.encode("utf-8")
        if not isinstance(out_s, str):
            out_s = out_s.decode("utf-8")
        print(out_s, end="", file=sys.stdout)

        if new_line:
            print()
        #sys.stdout.flush()


def add_summary(summary_writer, global_step, tag, value):
  """Add a new summary to the current summary_writer.
  Useful to log things that are not part of the training graph, e.g., tag=BLEU.
  """
  summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
  summary_writer.add_summary(summary, global_step)
