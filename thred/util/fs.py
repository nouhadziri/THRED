import errno
import os
import platform
import shutil
import subprocess
import shlex
import time
import pickle
import tarfile
import zipfile

from urllib.parse import urlparse


def split3(path):
    fld, f = os.path.split(path)
    fname, ext = os.path.splitext(f)

    return fld, fname, ext


def file_name(path):
    _, fname, _ = split3(path)
    return fname


def get_current_dir(path):
    if os.path.isdir(path):
        return path
    else:
        current_dir, _, _ = split3(path)
        return current_dir


def get_parent_dir(path):
    return os.path.abspath(os.path.join(get_current_dir(path), os.pardir))


def get_project_root_dir():
    return get_parent_dir(get_parent_dir(get_current_dir(__file__)))


def replace_ext(path, new_ext):
    dir, fname, ext = split3(path)
    return os.path.join(dir, "%s.%s" % (fname, new_ext))


def replace_dir(path, new_path, new_ext=None):
    _, fname, ext = split3(path)
    if new_ext is None:
        new_ext = ext
    else:
        new_ext = '.' + new_ext
    return os.path.join(new_path, fname + new_ext)


def mkdir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def rm_if_exists(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred


def is_url(url):
    return urlparse(url).scheme != ""


def copy(src, dst):
    shutil.copy(src, dst)


def rm_by_extension(working_dir, ext):
    matched_files = 0
    for f in os.listdir(working_dir):
        actual_f = os.path.join(working_dir, f)
        if f.endswith('.' + ext) and os.path.isfile(actual_f):
            os.remove(actual_f)
            matched_files += 1

    return matched_files


def count_lines(file_path):
    # https://gist.github.com/zed/0ac760859e614cd03652
    with open(file_path, 'rbU') as f:
        return sum(1 for _ in f)


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def uncompress(compressed_file, out_path="."):
    _, _, compressed_ext = split3(compressed_file)
    compressed_ext = compressed_ext.lower()

    if compressed_file.lower().endswith(".tar.gz") or compressed_ext in (".tgz", ".gz"):
        with tarfile.open(compressed_file, 'r:gz') as tgz_ref:
            tgz_ref.extractall(out_path)
    elif compressed_ext == ".zip":
        with zipfile.ZipFile(compressed_file, 'r') as zip_ref:
            zip_ref.extractall(out_path)
