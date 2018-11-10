import os

import tensorflow_hub as hub
import yaml
from gensim.models.keyedvectors import KeyedVectors
from redis.exceptions import ConnectionError

from util import fs
from util.kv import TinyRedis
from util.misc import Stopwatch


class GensimFactory:
    __cache = {}

    def __init__(self, vec_file):
        self._vec_file = vec_file

    def name(self):
        return 'gensim'

    def build(self, vocab_list):
        cache_key = (self._vec_file, len(vocab_list), vocab_list[-1])

        if cache_key in GensimFactory.__cache:
            print('  Embedding dict returned from the cache')
            return dict(GensimFactory.__cache[cache_key])

        pretrained_embeddings = KeyedVectors.load_word2vec_format(self._vec_file,
                                                                  encoding='iso-8859-1')

        embed_dict = {}

        for word in vocab_list:
            try:
                embed_dict[word] = pretrained_embeddings.wv[word]
            except KeyError:
                pass

        GensimFactory.__cache[cache_key] = embed_dict

        return dict(embed_dict)


class RedisFactory:
    def __init__(self, redis_port):
        self.__port = redis_port

    def name(self):
        return 'redis'

    def build(self, vocab_list):
        try:
            self.redis = TinyRedis(port=self.__port, max_connections=300)
            keyspace = self.redis.info('keyspace')
            if not keyspace:
                raise KeyError

            embed_dict = {}

            for word in vocab_list:
                vect = self.redis.lrange(word)
                if vect:
                    embed_dict[word] = [float(e) for e in vect]

            self.redis.close()

            return embed_dict
        except (ConnectionError, ConnectionRefusedError) as e:
            raise ValueError(e.args)
        except KeyError:
            raise ValueError('No keys found in redis, switched to gensim')


class HubFactory:
    __cache = {}

    def __init__(self, module_url):
        self.module_url = module_url

    def name(self):
        return 'TFHUB'

    def build(self, vocab_list, page_size=15000):
        cache_key = (self.module_url, len(vocab_list), vocab_list[-1])

        if cache_key in HubFactory.__cache:
            print('  Embedding dict returned from the cache')
            return dict(HubFactory.__cache[cache_key])

        print('  Loading TensorFlow HUB module...')
        os.environ["TFHUB_CACHE_DIR"] = ".tfhub_modules"
        embed = hub.Module(self.module_url)

        import tensorflow as tf

        embed_dict = {}

        num_pages = len(vocab_list) // page_size
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

            for i in range(num_pages + 1):
                lb = i * page_size
                ub = min((i + 1) * page_size, len(vocab_list))

                page = vocab_list[lb:ub]
                embedding_vectors = sess.run(embed(page))

                for i, word in enumerate(page):
                    if sum(embedding_vectors[i]) == 0:
                        continue

                    try:
                        embed_dict[word] = embedding_vectors[i]
                    except KeyError:
                        pass

        HubFactory.__cache[cache_key] = embed_dict
        return dict(embed_dict)


class WordEmbeddings:
    def __init__(self, config_path='conf/word_embeddings.yml'):
        with open(config_path, 'r') as file:
            self._args = yaml.load(file)

    def create_and_save(self, vocab_pkl_file, vocab_file, embedding_type, embedding_size, overwrite=False):
        if os.path.exists(vocab_pkl_file):
            if overwrite:
                os.remove(vocab_pkl_file)
            else:
                print('  vocab pickle already exists')
                return

        from util.vocab import load_vocab
        vocab_list, _ = load_vocab(vocab_file)
        embed_dict = self.create(embedding_type, embedding_size, vocab_list)
        fs.save_obj(embed_dict, vocab_pkl_file)

    def create(self, embedding_type, embedding_size, vocab_list):
        if embedding_type not in self._args:
            raise ValueError('Unsupported embedding type: ' + embedding_type)

        if '{}d'.format(embedding_size) not in self._args[embedding_type]:
            raise ValueError('Unsupported dimension size: ' + embedding_size)

        factory_dict = self._args[embedding_type]['{}d'.format(embedding_size)]

        sw = Stopwatch()

        embed_dict = {}
        if 'module_url' in factory_dict:
            embed_dict = HubFactory(factory_dict['module_url']).build(vocab_list)

        if not embed_dict and 'redis_port' in factory_dict:
            try:
                embed_dict = RedisFactory(int(factory_dict['redis_port'])).build(vocab_list)
            except ValueError as e:
                print('  Unable to use Redis: {}'.format(e))

        if not embed_dict and 'file' in factory_dict:
            embed_dict = GensimFactory(factory_dict['file']).build(vocab_list)

        sw.print('  Embedding dict created ({}/{} words exist)'.format(len(embed_dict), len(vocab_list)))

        return embed_dict
