from argparse import ArgumentParser
from pathlib import Path

from flair.data import Token

import tsv

parser = ArgumentParser(description='Predict segmentation on TSV')
parser.add_argument('model', help='path to model')
parser.add_argument('data', help='path to TSV file')
parser.add_argument('output', help='path to TSV file with predictions')

args = parser.parse_args()

columns = {0: 'text', 1: 'space_before', 2: 'tags', 3: 'poss', 4: 'year', 5: 'ambiguous', 6: 'label'}
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
    for i in range(1, len(sentence.tokens)):
        token: Token = sentence.tokens[i]
        token_before: Token = sentence.tokens[i - 1]
        if token.get_tag('space_before').value == '0':
            token_before.whitespace_after = False

from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger.load(args.model)
tagger.predict(train)

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
