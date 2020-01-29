"""Merges output form Morfeusz woth reference data."""
from argparse import ArgumentParser

import jsonlines
import regex

from ktagger import KText, KToken, KInterpretation

import morfeusz2


def morfeusz_tokenize(text: str, original_ktext: KText):
    ktext = KText(original_ktext.id)
    ktext.text = text
    ktext.year = original_ktext.year

    output = morfeusz.analyse(text)

    for start_position, end_position, i in output:
        form, pseudo_lemma, combined_tags, _, _ = i
        
        lemma = regex.sub(r':[abcdijnopqsv][0-9]?$', '', pseudo_lemma)
        
        kinterpretation = KInterpretation(lemma, combined_tags, disamb=False, manual=False)
        if ktext.tokens and ktext.tokens[-1].start_position == start_position and ktext.tokens[
            -1].end_position == end_position:
            ktext.tokens[-1].add_interpretation(kinterpretation)
        else:
            ktoken = KToken(form, space_before=None, start_offset=None, end_offset=None)
            ktoken.start_position = start_position
            ktoken.end_position = end_position
            ktoken.add_interpretation(kinterpretation)
            ktext.add_token(ktoken)
    return ktext


parser = ArgumentParser(description='Train')
parser.add_argument('jsonl_path', help='path to JSONL for getting text')
parser.add_argument('output_path', help='path to merged JSONL')
args = parser.parse_args()

morfeusz = morfeusz2.Morfeusz(expand_tags=True)  # dict_name=None, dict_path=None
#--dict-dir /home/kwrobel/repos/poleval2020-task2/data/ --dict morfeusz-f19

with jsonlines.open(args.jsonl_path) as reader, jsonlines.open(args.output_path, mode='w') as writer:
    for data in reader:
        original_ktext = KText.load(data)
        text = original_ktext.text

        ktext = morfeusz_tokenize(text, original_ktext)
        ktext.fix_offsets2()
        writer.write(ktext.save())
