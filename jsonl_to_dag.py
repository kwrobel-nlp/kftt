from argparse import ArgumentParser

import jsonlines

from ktagger import KText
from utils2 import jsonlines_gzip_reader

parser = ArgumentParser(description='Converts disamb JSONL to gold DAG')
parser.add_argument('disamb_path', help='path to disamb JSONL')
parser.add_argument('output_path', help='path to output DAG')
parser.add_argument('--sentences', action='store_true', help='split to sentences')
args = parser.parse_args()

with jsonlines_gzip_reader(args.disamb_path) as reader, open(args.output_path, 'w') as writer:
    for data in reader:
        ktext = KText.load(data)

        ktext.tokens = sorted(ktext.tokens, key=lambda t: (t.start_offset, t.end_offset))

        i = 0
        for token in ktext.tokens:
            # print()
            if not token.has_disamb():
                continue
            tags = set([interpretation.tag for interpretation in token.interpretations if
                        not interpretation.manual])
            poss = set([tag.split(':', 1)[0] for tag in tags])
            space = '1' if token.space_before else '0'



            writer.write('\t'.join([str(i), str(i + 1), token.form, 'X', token.disamb_tag(), space, 'disamb']))
            writer.write("\n")
            if 'ign' in tags:
                writer.write('\t'.join([str(i), str(i + 1), token.form, 'X', 'ign', space, '']))
                writer.write("\n")
            i += 1
            
            if args.sentences and token.sentence_end:
                writer.write("\n")
            
            # writer.write("\t".join(
            #     [token.form, space, '_'.join(sorted(tags)), '_'.join(sorted(poss)), str(ktext.year),
            #      token.disamb_tag()]))
        writer.write("\n")
