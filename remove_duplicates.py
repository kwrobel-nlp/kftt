import sys
from argparse import ArgumentParser

import jsonlines

from ktagger import KText

parser = ArgumentParser(description='Remove duplicates (keep first)')
parser.add_argument('input_path', help='path to JSONL')
parser.add_argument('output_path', help='path to output JSONL')
args = parser.parse_args()

reference_paragraphs = {}
with jsonlines.open(args.input_path) as reader:
    for data in reader:
        ktext = KText.load(data)
        text = ktext.text
        # text=ktext.infer_original_text()
        if text not in reference_paragraphs:
            reference_paragraphs[text] = ktext
        else:
            print('DUPLICATE', file=sys.stderr)
            print(text, file=sys.stderr)
            print(ktext.repr(), file=sys.stderr)
            print(reference_paragraphs[text].repr(), file=sys.stderr)
            print(file=sys.stderr)

with jsonlines.open(args.output_path, 'w') as writer:
    for ktext in reference_paragraphs.values():
        writer.write(ktext.save())
