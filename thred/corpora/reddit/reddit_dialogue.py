import codecs
from redis.exceptions import ResponseError

from thred.util import fs
from thred.util.misc import Stopwatch
from thred.util.kv import TinyRedis, install_redis, uninstall_redis


def build_lda_documents(reddit_text_path, reddit_db_path, output_path, lines_per_log=300000):
    sw = Stopwatch()

    print('start reading files...')

    docs = {}

    with codecs.getreader('utf-8')(open(reddit_text_path, 'rb')) as text_file, \
            open(reddit_db_path, 'r') as db_file:
        i = 0
        for line, reddit_text in zip(db_file, text_file):
            i += 1

            if i % lines_per_log == 0:
                sw.print('  {} lines processed'.format(i))

            reddit_text = reddit_text.strip()
            if not reddit_text:
                continue

            post = line.strip().split(',')
            post_type = int(post[0])
            if post_type == 0:
                link_id = post[3][3:]
            else:
                link_id = id

            if link_id in docs:
                docs[link_id].append(reddit_text)
            else:
                docs[link_id] = [reddit_text]

    sw.print('{} docs created...'.format(len(docs)))

    accepted_docs = 0
    with codecs.getwriter('utf-8')(open(output_path, 'wb')) as doc_file:
        for link_id, post_texts in docs.items():
            if len(post_texts) > 1:
                accepted_docs += 1
                doc_file.write('\t'.join(docs[link_id]) + '\n')

    sw.print("{}/{} docs written to output '{}'".format(accepted_docs, len(docs), output_path))


def build_conversational_data(reddit_text_path, reddit_db_path, output_path, redis_port, lines_per_log=100000):
    meta_redis = TinyRedis(port=redis_port, max_connections=10000)
    tree_redis = TinyRedis(port=redis_port + 1, max_connections=1000)

    # node_dict = {}

    sw = Stopwatch()
    print('start reading files...')
    with codecs.getreader('utf-8')(open(reddit_text_path, 'rb')) as text_file, \
            open(reddit_db_path, 'r') as db_file:
        i = 0
        pl = meta_redis.pipeline()
        for line, reddit_text in zip(db_file, text_file):
            i += 1

            if i % lines_per_log == 0:
                try:
                    pl.execute()
                except ResponseError:
                    print('  -- [error] waiting for bgsave to finish...')
                    while True:
                        try:
                            meta_redis.ping()
                            break
                        except ResponseError:
                            pass
                    print('  -- bgsave finished')
                    pl.execute()

                sw.print('  {} lines processed'.format(i))
                pl = meta_redis.pipeline()

            reddit_text = reddit_text.strip()
            if not reddit_text:
                continue

            post = line.strip().split(',')
            id = post[1]
            post_type = int(post[0])

            try:
                pl.hmset(id, {0: post_type,
                              1: reddit_text,
                              2: post[2],  # author
                              5: post[5],  # timestamp
                              6: post[6],  # subreddit
                              7: post[7],  # score
                              8: post[8],  # distinguished
                              9: post[9],  # gilded
                              10: post[10],  # controversiality
                              11: post[11],  # num_comments
                              12: post[12],  # num_crossposts
                              13: post[13],  # num_reports
                              })
            except IndexError:
                pl.hmset(id, {0: post_type,
                              1: reddit_text,
                              2: post[2],  # author
                              5: post[5],  # timestamp
                              6: post[6],  # subreddit
                              })

            tree_redis.sadd(id, id)
            # if id not in node_dict:
            #     new_node = Tree(id=id)
            #     node_dict[id] = new_node
            # else:
            #     new_node = node_dict[id]

            if post_type == 0:
                parent_id = post[4][3:]
                # link_id = post[3][3:]

                # tree_redis.sadd(parent_id, parent_id)
                tree_redis.set("p+{}".format(id), parent_id)
                tree_redis.sadd(parent_id, parent_id, id)
                # if parent_id in node_dict:
                #     parent_node = node_dict[parent_id]
                # else:
                #     parent_node = Tree(id=parent_id)
                #     node_dict[parent_id] = parent_node
                #
                # parent_node.add_child(new_node)

        data_size = i

    try:
        pl.execute()
    except ResponseError:
        print('  -- [error] waiting for bgsave to finish (processing already done) ...')
        while True:
            try:
                meta_redis.ping()
                break
            except ResponseError:
                pass
        print('  -- bgsave finished')
        pl.execute()
    print('generating output from trees...')

    generated_data_size = 0
    ids_path = fs.replace_ext(output_path, 'ids')
    with codecs.getwriter('utf-8')(open(output_path, 'wb')) as out_file, \
            open(ids_path, 'w') as ids_file:

        dialogues = []
        i = 0
        processed_roots = set()

        cursor = 0
        while True:
            cursor, keys = tree_redis.scan(cursor, count=10000)

            for key in keys:
                if key.startswith('p+'):
                    continue

                root = _get_root(key, tree_redis)
                if root in processed_roots:
                    continue

                dialogues.extend(_select_paths(_traverse_depth_first(root, tree_redis), meta_redis))
                processed_roots.add(root)

                i += 1
                if i % lines_per_log == 0:
                    generated_data_size += len(dialogues)
                    for reddit_ids, dialogue in dialogues:
                        out_file.write('\t'.join(dialogue) + '\n')
                        ids_file.write('\t'.join(reddit_ids) + '\n')

                    sw.print('  {} dialogues generated - {} so far'.format(len(dialogues), generated_data_size))
                    dialogues = []

            if cursor == 0:
                break

        generated_data_size += len(dialogues)
        for reddit_ids, dialogue in dialogues:
            out_file.write('\t'.join(dialogue) + '\n')
            ids_file.write('\t'.join(reddit_ids) + '\n')

    meta_redis.close()
    tree_redis.close()

    sw.print('Done. The dataset is built! {} generated out of {}'.format(generated_data_size, data_size))


def prepare_conversational_data(reddit_dialogue_path, num_turns, min_utterance_length, steps_per_flush=50000):
    assert num_turns >= 2

    output = fs.replace_ext(reddit_dialogue_path, '{}T'.format(num_turns) + '.txt')

    short_utterances, insufficient_turns = 0, 0
    with codecs.getreader("utf-8")(
            open(reddit_dialogue_path, mode="rb")) as data_file:
        with codecs.getwriter("utf-8")(open(output, mode="wb")) as out_file:
            print("Reading sets out...")

            lno = 0
            batch = []
            for line in data_file:
                lno += 1

                if lno % steps_per_flush == 0:
                    for conversation in batch:
                        out_file.write(conversation + '\n')
                    batch = []
                    print("  processed %d lines" % (lno))

                utterances = line.strip().split("\t")

                if len(utterances) - 1 < num_turns:
                    short_utterances += 1
                    # print("  line %d skipped (not enough turns: %d)" % (lno, len(utterances) - 1))
                    continue

                for i in range(len(utterances) - 1):
                    lb = i
                    ub = min(i + num_turns, len(utterances) - 1)

                    tokenized_tokens = []
                    too_short_utterance = False
                    for utter in utterances[lb:ub]:
                        if len(utter.split()) < min_utterance_length:
                            too_short_utterance = True
                            break
                        tokenized_tokens.append(utter)

                    if not too_short_utterance:
                        batch.append('\t'.join(tokenized_tokens))
                    else:
                        insufficient_turns += 1

                    if i >= len(utterances) - 1 - num_turns:
                        break

            for conversation in batch:
                out_file.write(conversation + '\n')

            print(
                "Done (%d lines | %d insufficient turns | %d short utterances)! But remove duplicates via sort | uniq commands" %
                (lno, insufficient_turns, short_utterances))


def _select_paths(root_to_leaf_paths, redis):
    accepted_paths = []
    for path in root_to_leaf_paths:
        if len(path) < 2:
            continue

        texts = [redis.hget(node, '1') for node in path]

        last_empty, empty_found = 0, False
        for i in range(len(path)):
            if not texts[i]:
                if i - last_empty >= 2:
                    accepted_paths.append((path[last_empty:i], texts[last_empty:i]))
                last_empty = i
                empty_found = True

        if not empty_found:
            accepted_paths.append((path, texts))
        elif len(path) - last_empty > 2:
            accepted_paths.append((path[last_empty + 1:], texts[last_empty + 1:]))

    return accepted_paths


def _get_root(id, tree_redis):
    n = id
    while True:
        p = tree_redis.get('p+{}'.format(n))
        if p is None or p == n:
            break
        n = p

    return n


def _traverse_depth_first(id, tree_redis):
    result = []

    nodes_to_visit = [(id, [id])]
    while nodes_to_visit:
        current_node, stack = nodes_to_visit.pop(0)

        children = tree_redis.smembers(current_node)
        for ch in children:
            if ch == current_node:
                continue

            new_stack = list(stack)
            new_stack.append(ch)
            nodes_to_visit.append((ch, new_stack))

        if len(children) <= 1:
            result.append(stack)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="mode")
    b_group = subparsers.add_parser("build")
    b_group.add_argument('-t', '--text_file', type=str, required=True, help='reddit text file')
    b_group.add_argument('-c', '--csv_file', type=str, required=True, help='reddit csv file')
    b_group.add_argument('-p', '--redis_port', type=int, default=7801, help='redis port (will use port+1 too)')
    b_group.add_argument('-o', '--output', type=str, required=True, help='output file')
    b_group.set_defaults(mode=lambda: "build")

    p_group = subparsers.add_parser("prepare")
    p_group.add_argument('-d', '--dialogue_data', type=str, required=True, help='reddit text file')
    p_group.add_argument('-t', '--num_turns', type=int, required=True, help='number of turns')
    p_group.add_argument('-l', '--min_length', type=int, default=2, help='minimum utterance length')
    p_group.set_defaults(mode=lambda: "prepare")

    lda_group = subparsers.add_parser("lda")
    lda_group.add_argument('-t', '--text_file', type=str, required=True, help='reddit text file')
    lda_group.add_argument('-c', '--csv_file', type=str, required=True, help='reddit csv file')
    lda_group.add_argument('-o', '--output', type=str, required=True, help='output file')
    lda_group.set_defaults(mode=lambda: "lda")

    params = parser.parse_args()

    if params.mode() == "build":
        redis1 = install_redis(port=params.redis_port, verbose=True)
        redis2 = install_redis(port=params.redis_port + 1, verbose=True)
        build_conversational_data(params.text_file, params.csv_file, params.output, params.redis_port)
        uninstall_redis(redis1)
        uninstall_redis(redis2)
    elif params.mode() == "lda":
        build_lda_documents(params.text_file, params.csv_file, params.output)
    else:
        prepare_conversational_data(params.dialogue_data, params.num_turns, params.min_length)
    # build_dataset_from_trees(dictionary)
