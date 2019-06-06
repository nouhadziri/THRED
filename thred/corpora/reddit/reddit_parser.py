import argparse
import codecs
import json
# import logging
import os
import re
import sre_constants
import sys
from collections import Counter

import collections
import mistune
import smart_open
from bs4 import BeautifulSoup
from spacy.lang.en import English

from .reddit_utils import RedditBotHandler
from thred.util.chartable import get_table
from thred.util.misc import Stopwatch
from thred.util.nlp import strip_emojis_and_emoticons

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# logger = logging.getLogger('reddit')

alphabet = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    ' '
]

nlp = English()
tokenizer = English().Defaults.create_tokenizer(nlp)


class __RedditPost(
    collections.namedtuple("RedditPost",
                           ("type", "id", "author", "link_id", "parent_id", "text", "created_utc", "subreddit_id",
                            "score", "distinguished", "gilded",
                            "controversiality",
                            "num_comments", "num_crossposts", "num_reports", "brand_safe", "url"))):
    pass


class __Subreddit:
    def __init__(self, id, name, fields=None, preprocess_cmd=None):
        self._id = id
        self._name = name

        self._fields = []
        if fields is not None:
            fields = fields.strip().split(',')
            for f in fields:
                if f.lower() not in ('title', 'selftext'):
                    raise ValueError('subreddit: field must be either title or selftext')
                self._fields.append(f.lower())

        if preprocess_cmd is None:
            self._affixes = None
        else:
            subcmd = preprocess_cmd.strip().split('$')
            self._positions = subcmd[0].split(',')
            self._affixes = sorted(subcmd[1].split(','), key=lambda s: len(s), reverse=True)
            self._new_affixes = None
            if len(subcmd) > 2:
                self._new_affixes = subcmd[2].split(',')
                if len(self._new_affixes) == 1:
                    self._new_affixes = [self._new_affixes[0]] * len(self._affixes)
                elif len(self._new_affixes) != len(self._affixes):
                    raise ValueError("No. Replace texts doesn't match with No. affixes")

    def generate_text(self, text_dict):
        if self._fields is None:
            return None

        text = '\n'.join([text_dict[f] for f in self._fields])

        if self._affixes:
            positions = set(self._positions)
            for i, affix in enumerate(self._affixes):
                if 'start' in positions and text.lower().startswith(affix.lower()):
                    if self._new_affixes:
                        lindex = text.lower().find(affix.lower())
                        text = self._new_affixes[i] + text[lindex + len(affix):]
                    else:
                        text = text[len(affix):].strip()

                    positions.remove('start')
                elif 'end' in positions and text.lower().endswith(affix.lower()):
                    if self._new_affixes:
                        rindex = text.lower().rfind(affix.lower())
                        text = text[:rindex] + self._new_affixes[i]
                    else:
                        text = text[:-len(affix)].strip()
                    positions.remove('end')

                if not positions:
                    break

        return text


def normalize_post_text(text, charmap=None):
    # footer
    # normalized = re.sub(r'(.+)\n---\n.+', r'\1', text, flags=re.DOTALL)

    # quotes
    normalized = re.sub(r'^&gt;.+\n?.+\n\n', '', text, flags=re.M)

    # derived from https://stackoverflow.com/a/761847
    # stripping markdown syntax
    html = mistune.markdown(normalized)
    normalized = ''.join(BeautifulSoup(html, "html5lib").findAll(text=True)).strip()

    # whitespace html entity
    normalized = re.sub(r'&nbsp;', ' ', normalized)

    # ^() syntax
    normalized = re.sub(r'\^\(([^\^]+)\)', r'\1', normalized)
    normalized = re.sub(r'(^|\s|\w+|[/,|])\^(\s|\w+|[/,|])', r'\1\2', normalized)

    normalized = re.sub(r'[\u2018\u2019]', "'", normalized.replace("\u2605", "*"))
    normalized = re.sub(r'[\u201C\u201D]', '"', normalized)

    # consecutive whitespaces
    normalized = re.sub(r'\s+', ' ', normalized)

    # users and subreddit mentions
    normalized = re.sub(r'/u/([\w\-]+)', 'RedditUser', normalized)
    normalized = re.sub(r'(^|\s)[@+]([\w\-]+(\s|:|$))', 'RedditUser', normalized)
    normalized = re.sub(r'(^|\s)/?\s?r\s?/([a-zA-Z0-9]\w+)(\s|:|\.|,|\?|$)', r'\1@url$\3', normalized)
    normalized = re.sub(r'(\w|\.|:)((http|ftp)s?://)', r'\1 \2', normalized)

    url_regex = r'\b((http|ftp)s?://)?(www\.)?[\w\-@:%_+~#=]+(\.[a-zA-Z]{2,3})+(/.*)?(\?.*)?\b'
    normalized = re.sub(url_regex, '@url$', normalized)

    normalized = strip_emojis_and_emoticons(normalized).strip()

    normalized = re.sub('´', "'", normalized)
    normalized = re.sub(r'-LRB-', '(', normalized, re.IGNORECASE)
    normalized = re.sub(r'-RRB-', ')', normalized, re.IGNORECASE)
    normalized = re.sub(r'[\u2500-\u25ff\u2800-\u28ff\u2700-\u27bf\u2000-\u204A]', '', normalized)
    normalized = re.sub(r'[\u0C81-\u0CF2\-][_.][\u0C81-\u0CF2\-]', '', normalized)
    normalized = re.sub(r'[\u0E81-\u0EDF\u30A0-\u30FF\u0300-\u0362\uFF5F-\uFFEE\u0C81-\u0CF2\u00AF-\u00B0'
                        r'\u0275-\u027B\u0292-\u0296\u02AC-\u02AF\u0298\u029A]', '', normalized)
    normalized = re.sub(r'[\u0D00-\u0D7F\u0E00-\u0E7F\u1780-\u17FF\u1400-\u167F]', '', normalized)
    s1, e1 = u"\U0001F100", u"\U0001F9FF"
    s2, e2 = u"\U00010000", u"\U0001342E"
    normalized = re.sub(r'[{}-{}{}-{}]'.format(s1, e1, s2, e2), '', normalized)
    normalized = re.sub(r"[\uFEFF\u2060\u2E18]", '', normalized)

    normalized = re.sub(r'(\.\.\.\s){2,}', '... ', normalized)
    normalized = re.sub(r'(([\^.></_|])\s){2,}', '', normalized)
    normalized = re.sub(r'(\"){2,}', '"', normalized)
    normalized = re.sub(r'(_){2,}', '', normalized)
    normalized = re.sub(r'!{2,}', '!!', normalized)

    if charmap is not None:
        for ch in charmap:
            if ch in normalized:
                try:
                    normalized = re.sub(ch, charmap[ch], normalized)
                except sre_constants.error:
                    pass

    normalized = re.sub(r'\([_\-.\u2200-\u22FF\s]+\)', '', normalized)
    normalized = re.sub(r'\[[_\-.\u2200-\u22FF\s]+\]', '', normalized)
    normalized = re.sub(r'\\[_\-.\u2200-\u22FF\s]+/', '', normalized)

    normalized = re.sub('¿', '', normalized)

    return normalized.strip()


def is_textual(text, threshold=0.85):
    counts = Counter(text)

    cnt = 0
    for letter in alphabet:
        if letter in counts:
            cnt += counts[letter]

    return cnt / len(text) >= threshold


def parse(input_data, params, convert_to_post):
    subreddits = _prepare_subreddits(params.subreddits)
    if subreddits:
        print('Subreddit whitelist provided with size {} '.format(len(subreddits)))

    # url_regex = r'((https?://)?|(ftps?://)?)(www\.)?[\w\-.@:%_+~#=]+(\.[a-zA-Z]{2,3})+(/.*)?(\?.*)?'

    batch_sw = Stopwatch()
    sw = Stopwatch()

    stats = {
        'processed': 0,
        'total': 0,
        'deleted_or_bot': 0,
        'not_in_subreddits': 0,
        'norm_empty': 0,
        'short_word_len': 0,
        'long_len': 0,
        'rt_err': 0,
        'not_en': 0,
    }

    charmap = get_table()

    bot_handler = RedditBotHandler()

    try:
        posts = []

        if params.skip_lines > 0:
            print('skipping {} lines...'.format(params.skip_lines))

        for line in input_data:
            try:
                line = line.decode()
            except AttributeError:
                pass

            stats['total'] += 1
            if stats['total'] <= params.skip_lines:
                continue

            if stats['total'] % params.batch_size == 0:
                print('@STAT {} lines / {} processed / long {} / norm_empty {} / short_words {}'
                      ' / sub {} / del,bot {} / not_en {} / rt_err {} / time {}s'.format(
                    stats['total'], stats['processed'],
                    stats['long_len'],
                    stats['norm_empty'], stats['short_word_len'],
                    stats['not_in_subreddits'], stats['deleted_or_bot'],
                    stats['not_en'],
                    stats['rt_err'],
                    batch_sw.elapsed()))
                batch_sw = Stopwatch()

                persist(params.out_dir, posts, params.output_prefix)

                posts = []

            json_post = json.loads(line)

            post = convert_to_post(json_post, subreddits)

            if not post:
                continue

            if subreddits and post.subreddit_id not in subreddits:
                stats['not_in_subreddits'] += 1
                continue

            if post.text == '[deleted]' or bot_handler.is_bot(post.author):
                stats['deleted_or_bot'] += 1
                continue

            try:
                normalized_text = normalize_post_text(post.text, charmap)

                if not normalized_text:
                    stats['norm_empty'] += 1
                    continue

                if params.max_words is None and len(normalized_text) > params.max_chars:
                    stats['long_len'] += 1
                    continue

                if not is_textual(normalized_text):
                    stats['not_en'] += 1
                    continue

                # tagged_words = corenlp.tokenize(normalized_text, client_id=stats['total'] % params.n_corenlps)
                tokens = tokenizer(normalized_text)
                tokens = [tk.text.lower() for tk in tokens]

                if not tokens:
                    stats['short_word_len'] += 1
                    continue

                if params.max_words is not None and len(tokens) > params.max_words:
                    stats['long_len'] += 1
                    continue
                if len(tokens) < params.min_words:
                    stats['short_word_len'] += 1
                    continue

                for i, tok in enumerate(tokens):
                    if tok == "@url$":
                        tokens[i] = '<url>'
                    if tok == 'reddituser':
                        tokens[i] = '<person>'
            except RuntimeError:
                stats['rt_err'] += 1
                continue

            post = post._replace(text=" ".join(tokens))

            posts.append(post)
            stats['processed'] += 1

        persist(params.out_dir, posts, params.output_prefix)
    except BaseException as e:
        print('Error occurred at {}: {}'.format(stats['total'], e), file=sys.stderr)
        if params.crash_file is not None:
            with open(params.crash_file, 'w') as crash_report:
                crash_report.write('{}'.format(stats['total'] - 1))
        raise e

    print('DONE!! {} lines / {} processed / long {} / norm_empty {} / short_words {}'
          ' / sub {} / del,bot {} / not_en {} / rt_err {} / time {}s'.format(
        stats['total'], stats['processed'],
        stats['long_len'],
        stats['norm_empty'], stats['short_word_len'],
        stats['not_in_subreddits'], stats['deleted_or_bot'],
        stats['not_en'], stats['rt_err'],
        sw.elapsed()))


def persist(out_dir, posts, file_name, save_text=True):
    csv_path = os.path.join(out_dir, '{}_db.csv'.format(file_name))
    txt_path = os.path.join(out_dir, '{}.txt'.format(file_name))

    txt_file = None
    if save_text:
        txt_file = codecs.open(txt_path, mode="a", encoding="utf-8", errors="ignore")

    with codecs.open(csv_path, mode="a", encoding="utf-8", errors="ignore") as csv_file:
        try:
            for post in posts:

                vals = []
                for field in post._fields:
                    if field == 'text':
                        continue
                    val = getattr(post, field)
                    vals.append(str(val) if val is not None else '')

                csv_file.write(','.join(vals) + "\n")

                if txt_file is not None:
                    txt_file.write('{}\n'.format(post.text))
        finally:
            if txt_file is not None:
                txt_file.close()

    sys.stdout.flush()


def _convert_comment_to_post(comment, subreddits):
    return __RedditPost(type=0, author=comment['author'],
                        id=comment['id'], link_id=comment['link_id'], parent_id=comment['parent_id'],
                        text=comment['body'],
                        created_utc=comment['created_utc'],
                        subreddit_id=comment['subreddit_id'],
                        score=comment['score'],
                        distinguished=comment['distinguished'],
                        gilded=comment['gilded'],
                        controversiality=comment['controversiality'],
                        num_comments=None,
                        num_crossposts=None,
                        num_reports=None,
                        brand_safe=None,
                        url='')


def _convert_submission_to_post(submission, subreddits):
    try:
        subreddit_id = submission['subreddit_id']
    except KeyError:
        return None

    post_text = ''
    if subreddit_id in subreddits:
        post_text = subreddits[subreddit_id].generate_text(
            {'title': submission['title'], 'selftext': submission['selftext']})
        if post_text is None:
            return None

    # if 'url' in submission:
    #     url = submission['url']
    # else:
    #     url = ''

    return __RedditPost(type=1, author=submission['author'],
                        id=submission['id'], link_id=submission['id'], parent_id='',
                        text=post_text,
                        created_utc=submission['created_utc'],
                        subreddit_id=subreddit_id,
                        score=submission['score'],
                        distinguished=submission['distinguished'],
                        controversiality=submission['suggested_sort'] if 'suggested_sort' in submission else None,
                        gilded=submission['gilded'],
                        num_comments=submission['num_comments'],
                        num_crossposts=None,
                        num_reports=None,
                        brand_safe=submission['brand_safe'] if 'brand_safe' in submission else None,
                        url=None)


def _prepare_subreddits(subreddits_arg):
    subreddits = {}

    if subreddits_arg is not None:
        with open(subreddits_arg, 'r') as subreddits_file:
            for subreddit_line in subreddits_file:
                parts = subreddit_line.split('\t')
                id, name = parts[1].strip(), parts[0].strip()
                subreddits[id] = __Subreddit(id, name,
                                             fields=parts[2] if len(parts) > 2 else None,
                                             preprocess_cmd=parts[3] if len(parts) > 3 else None)

    return subreddits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='output directory')
    parser.add_argument('-p', '--output_prefix', type=str, required=True,
                        help='prefix corresponding to output file name')

    parser.add_argument('-b', '--batch_size', type=int, default=100000, help='batch size')
    parser.add_argument('--min_chars', type=int, default=5,
                        help='minimum number of characters in a message (after normalization)')
    parser.add_argument('--max_chars', type=int, default=150,
                        help='maximum number of characters in a message (after normalization)')
    parser.add_argument('--max_words', type=int,
                        help='maximum number of words in a message (after normalization). '
                             'Disabled by default and overrides max_chars if specified.')
    parser.add_argument('--min_words', type=int, default=2,
                        help='minimum number of words in a message (after normalization)')
    parser.add_argument('-t', '--subreddits', type=str, help='list of accepted subreddits')

    parser.add_argument('--comments_file', help='reddit comment bz2 file')
    parser.add_argument('--submissions_file', help='reddit submission bz2 file')
    parser.add_argument('--comments_stream', action='store_true',
                        help='reddit comment stream (reading from pipe). Recommended for xz file')
    parser.add_argument('--submissions_stream', action='store_true',
                        help='reddit submission stream (reading from pipe). Recommended for xz file')

    parser.add_argument('-k', '--skip_lines', type=int, default=0, help='number of lines to skip')
    parser.add_argument('-r', '--crash_file', type=str,
                        help='file to store the last processed line in case of failure')

    params = parser.parse_args()
    if params.submissions_file is None and params.comments_file is None \
            and not params.comments_stream and not params.submissions_stream:
        print('one of the following args must be provided: '
              'comments_file, comments_stream, submissions_file, comments_stream')
        exit(1)
    elif params.submissions_file is not None and params.comments_file is not None:
        print('one of the following args must be specified: comments_file, submissions_file')
        exit(1)
    elif params.submissions_stream and params.comments_stream:
        print('one of the following args must be specified: comments_stream, submissions_stream')
        exit(1)
    elif (params.submissions_stream or params.comments_stream) and \
            (params.submissions_file is not None or params.comments_file is not None):
        print('Cannot specify stream and file args at the same time.')
        exit(1)

    comments_input, submissions_input = None, None
    if params.submissions_file:
        submissions_input = smart_open.smart_open(params.submissions_file)
    elif params.submissions_stream:
        submissions_input = sys.stdin
    elif params.comments_file:
        comments_input = smart_open.smart_open(params.comments_file)
    elif params.comments_stream:
        comments_input = sys.stdin

    if comments_input is not None:
        parse(comments_input, params, _convert_comment_to_post)

    if submissions_input is not None:
        parse(submissions_input, params, _convert_submission_to_post)
