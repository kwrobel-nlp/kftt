from argparse import ArgumentParser

import jsonlines

from ktagger import KText

parser = ArgumentParser(description='Convert JSONL to TSV (for training).')
parser.add_argument('merged_path', help='path to merged JSONL')
parser.add_argument('output_path', help='path to output TSV')
parser.add_argument('-s', action='store_true', help='split to sentences')
parser.add_argument('-c', action='store_true', help='split to corpora')
args = parser.parse_args()

if args.c:
    writers={}
else:
    writer=open(args.output_path, 'w')

with jsonlines.open(args.merged_path) as reader:
    for data in reader:
        ktext = KText.load(data)
        
        if args.c:
            corpus = ktext.id.split('▁')[1].split('_')[0]
            if corpus not in writers:
                writers[corpus]=open(args.output_path+'_'+corpus, 'w')
            writer=writers[corpus]

        tokens=ktext.tokens
        tokens = [t for t in tokens if t.has_disamb()]
        tokens = sorted(tokens, key=lambda t: (t.start_offset, t.end_offset))

        for token in tokens:
            # print()
            #if not token.has_disamb():
            #    continue
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

if args.c:
    for w in writers.values():
        w.close()
else:
    writer.close()