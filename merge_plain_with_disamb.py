"""Merges output form Morfeusz woth reference data."""
from argparse import ArgumentParser

import jsonlines

from ktagger import KText

parser = ArgumentParser(description='Train')
parser.add_argument('plain_path', help='path to plain JSONL')
parser.add_argument('disamb_path', help='path to disamb JSONL (reference)')
parser.add_argument('output_path', help='path to merged JSONL')
# parser.add_argument('--only_disamb', action='store_true', help='save only disamb versions of tokens and interpretations')
args = parser.parse_args()

reference_paragraphs={}
with jsonlines.open(args.disamb_path) as reader:
    for data in reader:
        ktext=KText.load(data)
        reference_paragraphs[ktext.id]=ktext
        
with jsonlines.open(args.plain_path) as reader, jsonlines.open(args.output_path, mode='w') as writer:
    for data in reader:
        ktext=KText.load(data)
        reference_ktext=reference_paragraphs[ktext.id]
        
        # fix_reference_offsets(reference_ktext, ktext.infer_original_text())
        reference_ktext.fix_offsets(ktext.infer_original_text())
        
        for token in reference_ktext.tokens:
            ktext.add_reference_token(token)
        writer.write(ktext.save())