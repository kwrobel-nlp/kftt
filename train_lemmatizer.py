import json
from argparse import ArgumentParser

import collections
from tqdm import tqdm

from ktagger import KText
from utils2 import jsonlines_gzip_reader

parser = ArgumentParser(description='Count statistics about lemmas.')
parser.add_argument('disamb_path', help='path to disamb JSONL (reference)')
parser.add_argument('output_path', help='path to JSON with stats')
args = parser.parse_args()

stats_form_tag_lemma = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
stats_lemma = collections.defaultdict(int)
with jsonlines_gzip_reader(args.disamb_path) as reader:
    for data in tqdm(reader):
        ktext = KText.load(data)
        for token in ktext.tokens:

            for interp in token.interpretations:
                if interp.disamb:
                    stats_form_tag_lemma[token.form][interp.tag][interp.lemma] += 1
                    stats_lemma[interp.lemma] += 1

# save to JSON
with open(args.output_path, 'w') as outfile:
    json.dump([stats_form_tag_lemma, stats_lemma], outfile, ensure_ascii=False)

print(len(stats_form_tag_lemma))
print(len(stats_lemma))

print(stats_form_tag_lemma['pokazuje'])
print(stats_form_tag_lemma['niż'])
print(stats_form_tag_lemma['niemający'])
print(stats_lemma['maić'])
print(stats_lemma['mieć'])
print(stats_lemma['mając'])
print(stats_form_tag_lemma['czem'])
