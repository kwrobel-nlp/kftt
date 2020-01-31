"""Merges output form Morfeusz woth reference data."""
import collections
from argparse import ArgumentParser
from typing import List, Tuple

import jsonlines

from ktagger import KText

parser = ArgumentParser(description='')
parser.add_argument('path', help='path to JSONL (plain or merged)')

args = parser.parse_args()


def shortest_path(ktext: KText):
    
    start_positions=collections.defaultdict(dict)
    for token in ktext.tokens:
        start_positions[token.start_position][token.end_position]=token

    last_position = 0
    tokens=[]
    for start_position, end_positions in sorted(start_positions.items()):
        if start_position!=last_position:
            continue
        max_end_position=max(end_positions.keys())
        token = end_positions[max_end_position]
        tokens.append(token)
        last_position=token.end_position


    for token in tokens:
        print(" ".join([token.form.replace(' ',''), 'X', '1', 'X']))
    print()


with jsonlines.open(args.path) as reader:
    for data in reader:
        ktext=KText.load(data)
        
        shortest_path(ktext)