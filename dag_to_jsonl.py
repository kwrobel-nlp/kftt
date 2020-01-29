# 1. read plain and merge somehow woth disamb

import glob
import json
import os
import sys
from argparse import ArgumentParser

import jsonlines as jsonlines

from ktagger import KInterpretation, KToken, KText

"""
Reads DAGs from Morfeusz PolEval output (disambiguated or not).
"""

TOKENS = 'tokens'
YEARS = 'years'
SEGMENT = 'segment'
LEMMA = 'lemma'
SPACE_BEFORE = 'space_before'
TAG = 'tag'
START_POSITION = 'start_position'
END_POSITION = 'end_position'
DISAMB = 'disamb'
INTERPRETATIONS = 'interpretations'
START_OFFSET = 'start_offset'
END_OFFSET = 'end_offset'


def read_dag(path):
    paragraphs = []
    paragraph = {TOKENS: []}
    years = None
    # token_end_positions={0:0}
    for line in open(path):
        line = line[:-1]
        if line == '':  # end of paragraph
            if paragraph[TOKENS]:
                paragraphs.append(paragraph)
            paragraph = {TOKENS: [], YEARS: years}
            continue

        fields = line.split('\t')

        if len(fields) == 1:
            years = fields[0][1:]
            paragraph[YEARS] = years
            continue
        else:
            try:
                start_position, end_position, segment, lemma, tag, nps, disamb = fields
                # disamb = disamb == 'disamb'
                assert disamb in ['', 'disamb', 'disamb_manual']
            except ValueError:
                start_position, end_position, segment, lemma, tag, nps = fields
                disamb = ''

            start_position = int(start_position)
            end_position = int(end_position)

            nps = nps == 'nps'
            space_before = not nps

            last_token = paragraph[TOKENS][-1] if paragraph[TOKENS] else None
            if last_token is not None and \
                    last_token[START_POSITION] == start_position and \
                    last_token[END_POSITION] == end_position:
                assert last_token[SEGMENT] == segment
                last_token[INTERPRETATIONS].append({LEMMA: lemma,
                                                    TAG: tag,
                                                    DISAMB: disamb})
            else:
                token = {START_POSITION: start_position,
                         END_POSITION: end_position,
                         SEGMENT: segment,
                         SPACE_BEFORE: space_before,
                         INTERPRETATIONS: [{LEMMA: lemma,
                                            TAG: tag,
                                            DISAMB: disamb}]}
                paragraph[TOKENS].append(token)
    return paragraphs


def is_disamb(token):
    return any(['disamb' in interpretation[DISAMB] for interpretation in token[INTERPRETATIONS]])


def dag_offsets(paragraph):
    text = original_text(paragraph)
    start_offsets = {}
    end_offsets = {0: 0}
    for token in paragraph[TOKENS]:
        start_position = token[START_POSITION]
        end_position = token[END_POSITION]
        # print(start_position, end_position)
        try:
            previous_end_offset = end_offsets[start_position]
            if token[SPACE_BEFORE]:
                previous_end_offset += 1
            start_offsets[start_position] = previous_end_offset

            token[START_OFFSET] = previous_end_offset

            if text[previous_end_offset:previous_end_offset + len(token[SEGMENT])] == token[SEGMENT]:
                end_offsets[end_position] = previous_end_offset + len(token[SEGMENT])
                token[END_OFFSET] = previous_end_offset + len(token[SEGMENT])
            else:  # manually corrected tokenization introducing space before
                # previous_offset += 1
                # offsets[end_position] = previous_offset + len(token[SEGMENT])
                print('OMITTING token with different space before', path,
                      text[previous_end_offset:previous_end_offset + len(token[SEGMENT])], token[SEGMENT],
                      file=sys.stderr)
                token[START_OFFSET] = None
                token[END_OFFSET] = None
        except KeyError:
            print('OMITTING node without incoming edges', token[SEGMENT], file=sys.stderr)
            token[START_OFFSET] = None
            token[END_OFFSET] = None

    # print(offsets.values())
    del end_offsets[0]
    # return start_offsets, end_offsets


def original_text(paragraph):
    """ Bierze pod uwagę pierwszą interpretację z danego węzła. Problem gdy analizator dodaje spacje, któ©ych nie ma."""
    strings = []
    last_position = 0
    for token in paragraph[TOKENS]:
        if token[START_POSITION] == last_position:
            if token[SPACE_BEFORE]:
                strings.append(' ')
            strings.append(token[SEGMENT])
            last_position = token[END_POSITION]

    return ''.join(strings)

def convert_to_ktagger(path):
    file_name = os.path.basename(path)
    paragraphs = read_dag(path)
    # print(path, len(paragraphs))

    for paragraph_index, paragraph in enumerate(paragraphs):
        if args.only_disamb:
            tokens = [token for token in paragraph[TOKENS] if is_disamb(token)]
            paragraph[TOKENS] = tokens

        paragraph_id = f"{corpus}▁{file_name}▁{paragraph_index}"
        ktext = KText(paragraph_id)
        years = paragraph[YEARS]
        year_feature = years[:2]
        ktext.year = year_feature

        text = original_text(paragraph)
        ktext.text = text

        dag_offsets(paragraph)

        for token in paragraph[TOKENS]:
            ktoken = KToken(token[SEGMENT], token[SPACE_BEFORE], token[START_OFFSET], token[END_OFFSET])
            ktext.add_token(ktoken)
            ktoken.start_position = token[START_POSITION]
            ktoken.end_position = token[END_POSITION]
            for interpretation in token[INTERPRETATIONS]:
                disamb = 'disamb' in interpretation[DISAMB]
                if args.only_disamb and not disamb:
                    continue
                manual = 'manual' in interpretation[DISAMB]
                kinterpretation = KInterpretation(interpretation[LEMMA], interpretation[TAG], disamb, manual)
                ktoken.add_interpretation(kinterpretation)

        assert text == ktext.infer_original_text()
        ktext.check_offsets()

        # print(ktext.save())

        payload = json.loads(ktext.save2())
        k = KText.load(payload)
        # print(k)

        # print(ktext.save())
        # print(k.save())
        assert ktext.save2() == k.save2()
        # print(payload)
        assert payload == ktext.save()
        yield ktext


parser = ArgumentParser(description='Train')
parser.add_argument('path', help='path pattern to directory with DAG data')
parser.add_argument('output_path', help='path JSONL output')
parser.add_argument('corpus_name', help='corpus name')
parser.add_argument('--only_disamb', action='store_true', help='save only disamb versions of tokens and interpretations')
args = parser.parse_args()

corpus = args.corpus_name

with jsonlines.open(args.output_path, mode='w') as writer:
    for path in sorted(glob.glob(args.path)):
        for ktext in convert_to_ktagger(path):
            writer.write(ktext.save())