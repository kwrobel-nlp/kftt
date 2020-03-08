from argparse import ArgumentParser

import jsonlines

from ktagger import KText

parser = ArgumentParser(
    description='Merge JSONLs analyzed (output form Morfeusz) with disamb (reference data). Keep disamb sentence ends. If a gold tag is missing then is set to MASK.')
parser.add_argument('plain_path', help='path to plain JSONL')
parser.add_argument('disamb_path', help='path to disamb JSONL (reference)')
parser.add_argument('output_path', help='path to merged JSONL')
args = parser.parse_args()

reference_paragraphs = {}
with jsonlines.open(args.disamb_path) as reader:
    for data in reader:
        ktext = KText.load(data)
        reference_paragraphs[ktext.id] = ktext

with jsonlines.open(args.plain_path) as reader, jsonlines.open(args.output_path, mode='w') as writer:
    for data in reader:
        ktext = KText.load(data)
        reference_ktext = reference_paragraphs[ktext.id]

        # fix_reference_offsets(reference_ktext, ktext.infer_original_text())
        reference_ktext.fix_offsets(ktext.infer_original_text())

        for token in reference_ktext.tokens:
            ktext.add_reference_token(token)

        # TODO sort tokens by position?
        #ktext.tokens = sorted(ktext.tokens, key=lambda t: (t.start_offset, t.end_offset))

        writer.write(ktext.save())
