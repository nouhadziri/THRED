import codecs
import os
import re

from tqdm import tqdm

from thred.util import fs


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--profanity_file', type=str, help='the profanity file')
    parser.add_argument('-f', '--data_file', type=str, required=True, help="the data file")

    params = parser.parse_args()

    if params.profanity_file is None:
        containing_dir, _, _ = fs.split3(os.path.abspath(__file__))
        profanity_file = os.path.join(containing_dir, "profanity_words.txt")
    else:
        profanity_file = params.profanity_file

    one_word_profanities = set()
    multi_word_profanities = set()
    with codecs.getreader('utf-8')(open(profanity_file, 'rb')) as profanity_reader:
        for line in profanity_reader:
            line = line.strip()
            if not line:
                continue

            if ' ' in line:
                multi_word_profanities.add(line)
            else:
                one_word_profanities.add(line)

    print("Profanity words loaded ({} one words/{} multi words)".format(len(one_word_profanities), len(multi_word_profanities)))

    output_file = fs.replace_ext(params.data_file, 'filtered.txt')
    prof1, profN = 0, 0
    with codecs.getreader('utf-8')(open(params.data_file, 'rb')) as data_file, \
            codecs.getwriter('utf-8')(open(output_file, 'wb')) as out_file:
        for line in tqdm(data_file):
            post = line.strip()
            utterances = post.split("\t")
            filtered = False
            for utterance in utterances:

                for word in utterance.split():
                    if word in one_word_profanities:
                        filtered = True
                        prof1 += 1
                        break

                for profanity in multi_word_profanities:
                    if profanity in utterance:
                        filtered = True
                        profN += 1
                        break

            if not filtered:
                s, e = u"\U0001F100", u"\U0001F9FF"
                s2, e2 = u"\U00010000", u"\U0001342E"
                post = re.sub(r'[{}-{}{}-{}]'.format(s, e, s2, e2), '', post)
                post = re.sub(r"[\uFEFF\u2060\u2E18]", '', post)
                line = "\t".join([" ".join(w.strip() for w in u.split()) for u in post.split("\t")])
                out_file.write(line + "\n")

    print("Filtered {} (One word {} / Multi word {})".format(prof1 + profN, prof1, profN))


if __name__ == '__main__':
    main()