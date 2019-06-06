import gzip
import math
import os
import random
import string
import subprocess
from signal import SIGTERM
from time import time


def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


def safe_div(dividend, divisor):
    return (dividend / divisor) if divisor != 0 else 0


def safe_mod(dividend, divisor):
    return (dividend % divisor) if divisor != 0 else 0


def generate_random_string(length=5):
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))


def escRegex(term):
    return term.replace('\\', '\\\\') \
        .replace('(', '\(').replace(')', '\)') \
        .replace('*', '\*').replace('+', '\+') \
        .replace('[', '\[').replace(']', '\]')


def kill_java_process(process_name):
    jps_cmd = subprocess.Popen('jps', stdout=subprocess.PIPE)
    jps_output = jps_cmd.stdout.read()
    pids = set()
    for jps_pair in jps_output.split(b'\n'):
        _p = jps_pair.split()
        if len(_p) > 1:
            if _p[1].decode() == process_name:
                pids.add(int(_p[0]))

    jps_cmd.wait()
    if pids:
        for pid in pids:
            os.kill(pid, SIGTERM)


def gunzip(gz_path):
    with gzip.open(gz_path, 'rb') as in_file:
        return in_file.read()


class Stopwatch:
    def __init__(self):
        self.start()

    def start(self):
        self.__start = time()

    def elapsed(self):
        return round(time() - self.__start, 3)

    def print(self, log_text):
        print(log_text, 'elapsed: {}s'.format(self.elapsed()))
