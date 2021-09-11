"""Merges output form Morfeusz woth reference data."""
import collections
import sys
from argparse import ArgumentParser

import jsonlines

from ktagger import KText
from utils2 import jsonlines_gzip_reader, jsonlines_gzip_writer

parser = ArgumentParser(description='Train')
parser.add_argument('merged_path', help='path to merged JSONL')
parser.add_argument('output_path', help='path to output TSV')
parser.add_argument('--eos', action='store_true', help='mark end of sentences')
args = parser.parse_args()


with jsonlines_gzip_reader(args.merged_path) as reader, open(args.output_path, 'w') as writer:
    for data in reader:
        ktext = KText.load(data)

        end_offsets_tags = collections.defaultdict(set)
        for token in ktext.tokens:
            if token.manual:
                continue
            tags = set([interpretation.tag for interpretation in token.interpretations if
                        not interpretation.manual])
            if token.end_offset is None:
                print('ERROR no token end offset', token.save(), file=sys.stderr)
                continue
            end_offsets_tags[token.end_offset].update(tags)

        reference_end_offsets = set([token.end_offset for token in ktext.tokens if token.has_disamb()])
        reference_eos_offsets = set(
            [token.end_offset for token in ktext.tokens if token.has_disamb() and token.sentence_end])

        ambiguous_end_offsets = ktext.find_ambiguous_end_offsets()
        # print(ambiguous_end_offsets)
        text=ktext.text
        last_offset = 0
        for end_offset, tags in sorted(end_offsets_tags.items()):
            segment = text[last_offset:end_offset]
            # print(last_offset, end_offset)
            last_offset += len(segment)
            if segment[0] == ' ':
                space = '1'
                segment = segment[1:]
            else:
                space = '0'
            eot = '1' if end_offset in reference_end_offsets else '0'
            if args.eos and end_offset in reference_eos_offsets: eot = '2'
            #TODO if token.sentence_end: eot='2'
            
            ambiguous = '1' if end_offset in ambiguous_end_offsets else '0'

            tags = end_offsets_tags[end_offset]
            poss = set([tag.split(':', 1)[0] for tag in tags])

            # print([segment, space, '_'.join(sorted(tags)), '_'.join(sorted(poss)), ktext.year, ambiguous, eot])
            writer.write("\t".join([segment, space, '_'.join(sorted(tags)), '_'.join(sorted(poss)), str(ktext.year), ambiguous, eot]))
            writer.write("\n")
        writer.write("\n")