import collections
import logging
import codecs
from pathlib import Path
from typing import Type, TypeVar, List
from os import environ, rename

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import yaml
from pymagnitude import Magnitude

from . import fs, wget, vocab
from .misc import Stopwatch

T = TypeVar('T')
logger = logging.getLogger(__name__)


class EmbeddingType(
    collections.namedtuple("EmbeddingType", ("name", "url", "dim", "src_type"))):
    pass


class EmbeddingFactory:
    def __init__(self, embedding_type: EmbeddingType):
        self._embedding_type = embedding_type

    def build(self, vocab_list: List[str], h5_file: str, **kwargs) -> (List[str], List[str]):
        """
        :param vocab_list: The vocabulary list built upon the dataset
        :param h5_file: The output h5 file where the embedding vectors will be saved into
        :param kwargs: Additional parameters based on the factory type
        :return A tuple containing the Out-Of-Vocabulary words and In-Vocabulary words
        """
        pass


class RandomFactory(EmbeddingFactory):
    def __init__(self, embedding_type: EmbeddingType):
        super().__init__(embedding_type)

    def build(self, vocab_list: List[str], h5_file: str, **kwargs) -> (List[str], List[str]):
        init_weight = kwargs.get('init_weight', 0.1)

        with h5py.File(h5_file, mode="w") as vec_h5:
            for w in vocab_list:
                vec = np.random.uniform(-init_weight, init_weight, size=self._embedding_type.dim)
                vec_h5.create_dataset("{key}/vec".format(key=w), data=vec)
                vec_h5.create_dataset("{key}/trainable".format(key=w), data=1)

        return vocab_list, []


class MagnitudeFactory(EmbeddingFactory):
    def __init__(self, embedding_type: EmbeddingType):
        super().__init__(embedding_type)

        cache_dir = Path(fs.get_project_root_dir()) / ".magnitude"
        fs.mkdir_if_not_exists(cache_dir)
        embed_file = self._embedding_type.url[self._embedding_type.url.rfind("/") + 1:]
        compressed_file = Path(cache_dir) / embed_file
        if not compressed_file.exists():
            logger.info('  Downloading magnitude file ("{}")...'.format(embed_file))
            wget.download(self._embedding_type.url, compressed_file)

        self._embed_file = compressed_file
        logger.info('  Loading Magnitude module...')
        self._magnitude_vecs = Magnitude(self._embed_file)

    def build(self, vocab_list: List[str], h5_file: Path, **kwargs) -> (List[str], List[str]):
        oov, iov = [], []
        with h5py.File(h5_file, mode="w") as vec_h5:
            for w in vocab_list:
                is_oov = w not in self._magnitude_vecs
                vec = self._magnitude_vecs.query(w)
                vec_h5.create_dataset("{key}/vec".format(key=w), data=vec)
                vec_h5.create_dataset("{key}/trainable".format(key=w), data=1 if is_oov else 0)

                if is_oov:
                    oov.append(w)
                else:
                    iov.append(w)

        return oov, iov


class TfHubFactory(EmbeddingFactory):
    __cache = {}

    def __init__(self, embedding_type: EmbeddingType):
        super(TfHubFactory, self).__init__(embedding_type)

    def build(self, vocab_list: List[str], h5_file: Path, **kwargs) -> (List[str], List[str]):
        page_size = kwargs.get('page_size', 15000)
        init_weight = kwargs.get('init_weight', 0.1)

        logger.info('  Loading TensorFlow HUB module...')
        environ["TFHUB_CACHE_DIR"] = ".tfhub_modules"
        embedder = hub.Module(self._embedding_type.url)

        num_pages = len(vocab_list) // page_size
        oov, iov = [], []
        with h5py.File(h5_file, mode='w') as vec_h5:
            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

                for i in range(num_pages + 1):
                    lb = i * page_size
                    ub = min((i + 1) * page_size, len(vocab_list))

                    page = vocab_list[lb:ub]
                    embedding_vectors = sess.run(embedder(page))

                    for i, word in enumerate(page):
                        is_oov = sum(embedding_vectors[i]) == 0
                        if is_oov:
                            vec = np.random.uniform(-init_weight, init_weight, self._embedding_type.dim)
                        else:
                            vec = embedding_vectors[i]

                        vec_h5.create_dataset("{key}/vec".format(key=word), data=vec)
                        vec_h5.create_dataset("{key}/trainable".format(key=word), data=1 if is_oov else 0)

                        if is_oov:
                            oov.append(word)
                        else:
                            iov.append(word)

        return oov, iov


class EmbeddingUtil:
    def __init__(self, config_path: str='conf/word_embeddings.yml'):
        with open(config_path, 'r') as file:
            self._args = yaml.load(file)

    @classmethod
    def from_type(cls: Type[T], embedding_type: EmbeddingType) -> EmbeddingFactory:
        if embedding_type.src_type == "tfhub":
            return TfHubFactory(embedding_type)
        elif embedding_type.src_type == "magnitude":
            return MagnitudeFactory(embedding_type)
        elif embedding_type.src_type == "random":
            return RandomFactory(embedding_type)
        else:
            raise ValueError(
                "Unknown source type '{}' defined in the embedding config file".format(embedding_type.src_type))

    @classmethod
    def load_vectors(cls: Type[T], vocab_h5: str, vocab_file: str) -> (np.ndarray, np.ndarray, np.ndarray):
        vocab_list, _ = vocab.load_vocab(vocab_file)

        reserved_words, trainables, frozens = [], [], []
        with h5py.File(vocab_h5, mode='r') as vec_h5:
            for w in vocab_list:
                vec = vec_h5["{key}/vec".format(key=w)][...]
                is_trainable = vec_h5["{key}/trainable".format(key=w)][...]
                if w in vocab.RESERVED_WORDS:
                    reserved_words.append(vec)
                if is_trainable:
                    trainables.append(vec)
                else:
                    frozens.append(vec)

        return np.asarray(reserved_words), np.asarray(trainables), np.asarray(frozens)

    def build_if_not_exists(self, embedding_type: str, vocab_h5: str, vocab_file: str, overwrite: bool=False):
        if Path(vocab_h5).exists() and not overwrite:
            return

        sw = Stopwatch()

        if embedding_type.lower().startswith("random"):
            try:
                dim = int(embedding_type[len("random"):])
            except ValueError:
                dim = 300
                logger.warning("Unrecognizable dimension for random embedding. Set to default: {}".format(dim))
            _embed_type = EmbeddingType(embedding_type, "", dim, "random")
        else:
            e = self._args[embedding_type]
            _embed_type = EmbeddingType(embedding_type, e["url"], e["dim"], e["src_type"])

        vocab_list, _ = vocab.load_vocab(vocab_file)
        oov, iov = EmbeddingUtil.from_type(_embed_type).build(vocab_list, vocab_h5)

        rename(vocab_file, fs.replace_ext(vocab_file, 'tf'))
        with codecs.getwriter("utf-8")(open(vocab_file, "wb")) as writer:
            for rw in vocab.RESERVED_WORDS:
                writer.write("{}\n".format(rw))

            for w in oov:
                if w not in vocab.RESERVED_WORDS:
                    writer.write("{}\n".format(w))

            for w in iov:
                if w not in vocab.RESERVED_WORDS:
                    writer.write("{}\n".format(w))

        logger.info("Embedding vectors built from {} in {:.1f}s".format(embedding_type, sw.elapsed()))
