""" Key-Value store (i.e., Redis) utilities
"""
import os
import subprocess

import redis


class TinyRedis:
    """ TinyRedis is a wrapper on Redis functions and supports a subset of Redis functions.
        Upon construction, it pings the server, meaning that an execption would be thrown in case of failure.
    """
    def __init__(self, port, max_connections, host='localhost'):
        self.__r = redis.Redis(host=host, port=port, max_connections=max_connections, decode_responses=True)
        self.ping()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def pipeline(self):
        return self.__r.pipeline(transaction=False)

    def ping(self):
        self.__r.ping()

    def exists(self, key):
        return self.__r.exists(key)

    def delete(self, key):
        self.__r.delete(key)

    def set(self, key, value):
        return self.__r.set(key, value)

    def get(self, key):
        return self.__r.get(key)

    def hscan(self, key):
        return self.__r.hscan_iter(key)

    def hget(self, key, field):
        return self.__r.hget(key, field)

    def hmget(self, key, *fields):
        return self.__r.hmget(key, fields)

    def pl_hincrby(self, key, mappings):
        p = self.__r.pipeline(transaction=False)
        for field, amount in mappings.items():
            p.hincrby(key, field, amount)
        p.execute()

    def lrange(self, key, start_index=0, stop_index=-1):
        return self.__r.lrange(key, start=start_index, end=stop_index)

    def sadd(self, key, *members):
        return self.__r.sadd(key, *members)

    def smembers(self, key):
        return self.__r.smembers(key)

    def info(self, section=None):
        return self.__r.info(section)

    def pfadd(self, key, *elements):
        return self.__r.pfadd(key, *elements)

    def pfcount(self, key):
        return self.__r.pfcount(key)

    def scan(self, cursor=0, match=None, count=None):
        return self.__r.scan(cursor, match, count)

    def close(self):
        del self.__r


def install_redis(install_path='',
                  download_url='http://download.redis.io/releases/redis-5.0.3.tar.gz',
                  port=6384, verbose=True):
    import tarfile
    import tempfile
    from urllib.error import URLError, HTTPError
    import urllib.request as url_request

    from redis.exceptions import ConnectionError
    from .fs import split3

    proceed_install = True
    r = redis.Redis(host='localhost', port=port)
    try:
        r.ping()
        proceed_install = False
    except ConnectionError:
        pass

    if proceed_install:
        if not install_path:
            tmp_dir = tempfile.mkdtemp(prefix='redis{}'.format(port))
            install_path = os.path.join(tmp_dir, 'redis')

        if not os.path.exists(install_path):
            working_dir, redis_name, _ = split3(install_path)
            redis_tgzfile = os.path.join(working_dir, 'redis.tar.gz')

            if verbose:
                print('Downloading Redis...')

            try:
                with url_request.urlopen(download_url) as resp, \
                        open(redis_tgzfile, 'wb') as out_file:
                    data = resp.read()  # a `bytes` object
                    out_file.write(data)
            except HTTPError as e:
                if verbose:
                    if e.code == 404:
                        print(
                            'The provided URL seems to be broken. Please find a URL for Redis')
                    else:
                        print('Error code: ', e.code)
                raise ValueError(e)
            except URLError as e:
                if verbose:
                    print('URL error: ', e.reason)
                raise ValueError(e)

            if verbose:
                print('Extracting Redis...')
            with tarfile.open(redis_tgzfile, 'r:gz') as tgz_ref:
                tgz_ref.extractall(working_dir)

            os.remove(redis_tgzfile)

            redis_dir = None
            for f in os.listdir(working_dir):
                if f.lower().startswith('redis'):
                    redis_dir = os.path.join(working_dir, f)
                    break

            if not redis_dir:
                raise ValueError()

            os.rename(redis_dir, os.path.join(working_dir, redis_name))

            if verbose:
                print('Installing Redis...')

            redis_conf_file = os.path.join(install_path, 'redis.conf')
            subprocess.call(
                ['sed', '-i', 's/tcp-backlog [0-9]\+$/tcp-backlog 3000/g', redis_conf_file])
            subprocess.call(
                ['sed', '-i', 's/daemonize no$/daemonize yes/g', redis_conf_file])
            subprocess.call(
                ['sed', '-i', 's/pidfile .*\.pid$/pidfile redis_{}.pid/g'.format(port), redis_conf_file])
            subprocess.call(
                ['sed', '-i', 's/port 6379/port {}/g'.format(port), redis_conf_file])
            subprocess.call(
                ['sed', '-i', 's/save 900 1/save 15000 1/g', redis_conf_file])
            subprocess.call(
                ['sed', '-i', 's/save 300 10/#save 300 10/g', redis_conf_file])
            subprocess.call(
                ['sed', '-i', 's/save 60 10000/#save 60 10000/g', redis_conf_file])
            subprocess.call(
                ['sed', '-i', 's/logfile ""/logfile "redis_{}.log"/g'.format(port), redis_conf_file])
            subprocess.call(['make'], cwd=install_path)

        if verbose:
            print('Running Redis on port {}...'.format(port))
        subprocess.call(['src/redis-server', 'redis.conf'], cwd=install_path)

    return install_path


def uninstall_redis(install_path, verbose=True):
    import shutil

    if not install_path or not os.path.exists(install_path):
        if verbose:
            print("Cannot uninstall because the installation path does not exist!")
        return

    redis_conf_file = os.path.join(install_path, 'redis.conf')
    p = subprocess.Popen(['grep', '-E', '^port [0-9]+$', redis_conf_file], stdout=subprocess.PIPE)
    out, _ = p.communicate()

    port = out.split()[1].decode('utf-8')

    if verbose:
        print("Shutting down Redis on port {}...".format(port))

    subprocess.call(['src/redis-cli', '-p', port, 'shutdown'], cwd=install_path)

    shutil.rmtree(install_path)

    if verbose:
        print("Redis on port {} uninstalled...".format(port))
