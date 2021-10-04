from argparse import ArgumentParser
from pathlib import Path

from flair.data import Token, Sentence

import tsv
from jsonl_to_tsv_segmentation_every_char2 import set_space_after

parser = ArgumentParser(description='Predict segmentation on TSV')
parser.add_argument('model', help='path to model')
parser.add_argument('data', help='path to TSV file')
parser.add_argument('output', help='path to TSV file with predictions')
parser.add_argument('--mini_batch_size', default=32, type=int, help='mini batch size')

args = parser.parse_args()

columns = {0: 'text', 1: 'space_before', 2: 'tags', 3: 'poss', 4: 'year', 5: 'ambiguous'}
train = tsv.TSVDataset(
    Path(args.data),
    columns,
    tag_to_bioes=None,
    comment_symbol=None,
    in_memory=True,
    encoding="utf-8",
    document_separator_token=None
)



for sentence in train:
    set_space_after(sentence)

from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger.load(args.model)

import time
start_time = time.time()
tagger.predict(train, mini_batch_size=args.mini_batch_size)
print("--- %s seconds ---" % (time.time() - start_time))

with open(args.output, 'w') as writer:
    for sentence in train:
        for token in sentence.tokens:
            # token: Token = sentence.tokens[i]
            pred = token.get_tag('label').value
            score = token.get_tag('label').score
            # print(' '.join([token.text, 'X', pred, str(score)]))
            writer.write(' '.join([token.text, 'X', pred, str(score)]))
            writer.write('\n')
        writer.write('\n')
        # print()
