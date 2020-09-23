import collections
from argparse import ArgumentParser

import jsonlines

from ktagger import KText


def shortest_path(ktext: KText, longest=False):
    start_positions = collections.defaultdict(dict)
    for token in ktext.tokens:
        if token.manual: continue
        start_positions[token.start_position][token.end_position] = token

    last_position = 0
    tokens = []
    for start_position, end_positions in sorted(start_positions.items()):
        if start_position != last_position:
            continue
        if longest:
            max_end_position = min(end_positions.keys())
        else:
            max_end_position = max(end_positions.keys())
        token = end_positions[max_end_position]
        tokens.append(token)
        last_position = token.end_position

    return tokens


if __name__ == "__main__":
    parser = ArgumentParser(description='JSONL to DAG shortest or longest path')
    parser.add_argument('path', help='path to JSONL (plain or merged)')
    parser.add_argument('--longest', action='store_true', help='longest path')

    args = parser.parse_args()

    with jsonlines.open(args.path) as reader:
        for data in reader:
            ktext = KText.load(data)

            tokens = shortest_path(ktext, longest=args.longest)

            for token in tokens:
                print(" ".join([token.form.replace(' ', ''), 'X', '1', 'X']))
            print()
