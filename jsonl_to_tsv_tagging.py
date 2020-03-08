from argparse import ArgumentParser

import jsonlines

from ktagger import KText

parser = ArgumentParser(description='Convert JSONL to TSV (for training).')
parser.add_argument('merged_path', help='path to merged JSONL')
parser.add_argument('output_path', help='path to output TSV')
parser.add_argument('-s', action='store_true', help='split to sentences')
args = parser.parse_args()

with jsonlines.open(args.merged_path) as reader, open(args.output_path, 'w') as writer:
    for data in reader:
        ktext = KText.load(data)

        ktext.tokens = sorted(ktext.tokens, key=lambda t: (t.start_offset, t.end_offset))

        for token in ktext.tokens:
            # print()
            if not token.has_disamb():
                continue
            tags = set([interpretation.tag for interpretation in token.interpretations if
                        not interpretation.manual])
            poss = set([tag.split(':', 1)[0] for tag in tags])
            space = '1' if token.space_before else '0'

            writer.write("\t".join(
                [token.form, space, '_'.join(sorted(tags)), '_'.join(sorted(poss)), str(ktext.year),
                 token.disamb_tag()]))
            writer.write("\n")

            if args.s and token.sentence_end is True:
                writer.write("\n")
        writer.write("\n")
