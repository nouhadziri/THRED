from __future__ import print_function
import sys
import yaml
import codecs
from os import path, listdir, remove
from urllib.error import URLError, HTTPError

from gensim.scripts import glove2word2vec
from util.fs import mkdir_if_not_exists, rm_if_exists, split3, uncompress, replace_ext
from util import wget


def validate(args, embed_args):
    if args.embedding_file is not None:
        if not path.exists(args.embedding_file):
            raise ValueError("Cannot find the embedding file in the path.")
    elif args.embedding_type.lower() not in embed_args:
        raise ValueError("No entry '{}' in 'conf/word_embeddings.yml'. "
                         "If you want to use an embedding not listed in the yaml file, "
                         "you need to provide an embedding file (run the program using '-f').")
    elif "{}d".format(args.dimensions) not in embed_args[args.embedding_type.lower()]:
        raise ValueError("No entry '{}d' found for '{}' in 'conf/word_embeddings.yml'. "
                         "If you want to use an embedding not listed in the yaml file, "
                         "you need to provide an embedding file (run the program using '-f').")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embedding_type", default="glove", help="Embedding type. "
                                                                        "The default embeddings are 'glove' and 'word2vec' (see conf/word_embeddings.yml)."
                                                                        "However, other types can be added too.")
    parser.add_argument("-d", "--dimensions", default=300, type=int, help="#dimensions of Embedding vectors")
    parser.add_argument("-f", "--embedding_file",
                        help="Path to a compressed embedding file. "
                             "Gensim would be used to load the embeddings.")

    args = parser.parse_args()

    with open("conf/word_embeddings.yml", 'r') as file:
        embed_args = yaml.load(file)

    try:
        validate(args, embed_args)
    except ValueError as e:
        print(e, file=sys.stderr)
        exit(1)

    embed_key = args.embedding_type.lower()
    dim_key = '{}d'.format(args.dimensions)

    if args.embedding_file is not None:
        if embed_key not in embed_args:
            embed_args[embed_key] = {}

        if dim_key not in embed_args[embed_key]:
            embed_args[embed_key][dim_key] = { "file": path.abspath(args.embedding_file) }
    else:
        if "url" not in embed_args[embed_key][dim_key]:
            print("Seems no need to download any embedding files!!!")
            exit(0)

        containing_dir, _, _ = split3(path.abspath(__file__))
        workspace_dir = path.abspath(path.join(containing_dir, path.pardir, "workspace"))
        mkdir_if_not_exists(workspace_dir)
        embed_dir = path.join(workspace_dir, "embeddings")
        mkdir_if_not_exists(embed_dir)

        uncompressed_file = None
        download_url = embed_args[embed_key][dim_key]["url"]
        embed_file = download_url[download_url.rfind("/") + 1:]
        compressed_file = path.join(embed_dir, embed_file)

        if not path.exists(compressed_file):
            download_embeddings(download_url, compressed_file)

        print("Uncompressing the file...")
        uncompress(compressed_file, embed_dir)
        remove(compressed_file)

        for f in listdir(embed_dir):
            if (f.endswith(".txt") or f.endswith(".bin")) and str(args.dimensions) in f:
                uncompressed_file = path.join(embed_dir, f)
                break

        if embed_key == "glove":
            vec_file = replace_ext(uncompressed_file, 'vec')
            if not path.exists(vec_file):
                print("Converting to gensim understandable format...")
                glove2word2vec.glove2word2vec(uncompressed_file, vec_file)

            rm_if_exists(uncompressed_file)
            uncompressed_file = vec_file

        embed_args[embed_key][dim_key]["file"] = uncompressed_file

    with codecs.getwriter("utf-8")(open("conf/word_embeddings.yml", "wb")) as f:
        yaml.dump(embed_args, f, default_flow_style=True)

    print("Done. Config file successfully updated.")


def download_embeddings(download_url, out_file):
    try:
        wget.download(download_url, out_file)
    except HTTPError as e:
        if e.code == 404:
            raise ValueError("HttpError: The URL ({}) seems to be broken. "
                             "Please replace the URL in 'conf/word_embeddings.yml'.".format(download_url))
        else:
            raise ValueError("HttpError: error code {}".format(e.code))
    except URLError as e:
        raise ValueError(e)


if __name__ == "__main__":
    main()
