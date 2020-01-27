#1. read plain and merge somehow woth disamb

import glob
import os
import sys
from argparse import ArgumentParser
import collections

"""
Do nauki tokenizera,
TODO czy taggera też? nie, bo inna tokenizacja. tagger powinine się uczyć na stokenizowanym poprawnie
"""

SEGMENT = 'segment'
LEMMA = 'lemma'
SPACE_BEFORE = 'space_before'
TAG = 'tag'
END_POSITION = 'end_position'
START_POSITION = 'start_position'
TOKEN_END_POSITION = 'token_end_position'
TOKEN_START_POSITION = 'token_start_position'


def read_dag(path):
    paragraphs = []
    paragraph = {'tokens': []}
    years = None
    # token_end_positions={0:0}
    for line in open(path):
        line = line[:-1]
        if line == '':  # end of paragraph
            if paragraph['tokens']:
                paragraphs.append(paragraph)
            paragraph = {'tokens': [], 'years': years}
            continue

        fields = line.split('\t')

        if len(fields) == 1:
            years = fields[0][1:]
            paragraph['years'] = years
            continue
        else:
            try:
                start_position, end_position, segment, lemma, tag, nps, disamb = fields
                # disamb = disamb == 'disamb'
                assert disamb in ['', 'disamb', 'disamb_manual']
            except:
                start_position, end_position, segment, lemma, tag, nps = fields
                disamb = ''

            start_position = int(start_position)
            end_position = int(end_position)

            nps = nps == 'nps'
            space_before = not nps

            last_token = paragraph['tokens'][-1] if paragraph['tokens'] else None
            if last_token is not None and \
                    last_token[START_POSITION] == start_position and \
                    last_token[END_POSITION] == end_position:
                assert last_token[SEGMENT] == segment
                last_token['interpretations'].append({LEMMA: lemma,
                                                      TAG: tag,
                                                      'disamb': disamb})
            else:
                token = {START_POSITION: start_position,
                         END_POSITION: end_position,
                         SEGMENT: segment,
                         SPACE_BEFORE: space_before,
                         'interpretations': [{LEMMA: lemma,
                                              TAG: tag,
                                              'disamb': disamb}]}
                paragraph['tokens'].append(token)
    return paragraphs

def is_disamb(token):
    return any(['disamb' in interpretation['disamb'] for interpretation in token['interpretations']])

def dag_offsets(paragraph, disamb_only=False, path=None):
    text = original_text(paragraph)
    # print(text)
    offsets = {0: 0}
    for token in paragraph['tokens']:
        if disamb_only and not is_disamb(token):
            continue
        start_position = token['start_position']
        end_position = token['end_position']
        # print(start_position, end_position)
        try:
            previous_offset = offsets[start_position]
            if token[SPACE_BEFORE]:
                previous_offset += 1
    
            if text[previous_offset:previous_offset + len(token[SEGMENT])] == token[SEGMENT]:
                offsets[end_position] = previous_offset + len(token[SEGMENT])
            else:  # manually corrected tokenization introducing space before
                # previous_offset += 1
                # offsets[end_position] = previous_offset + len(token[SEGMENT])
                print('OMITTING token with different space before', path, text[previous_offset:previous_offset + len(token[SEGMENT])], token[SEGMENT],
                      file=sys.stderr)
        except KeyError:
            print('OMITTING node without incoming edges', token[SEGMENT], file=sys.stderr)
            
    # print(offsets.values())
    del offsets[0]
    return offsets



def original_text(paragraph):
    """ Bierze pod uwagę pierwszą interpretację z danego węzła. Problem gdy analizator dodaje spacje, któ©ych nie ma."""
    strings = []
    last_position=0
    for token in paragraph['tokens']:
        if token[START_POSITION]==last_position:
            if token[SPACE_BEFORE]:
                strings.append(' ')
            strings.append(token[SEGMENT])
            last_position=token[END_POSITION]
            
    return ''.join(strings)

parser = ArgumentParser(description='Train')
parser.add_argument('path', help='path pattern to directory with plain data')
parser.add_argument('--disamb', default=None, help='path to directory with disamb data')
args = parser.parse_args()

for path in sorted(glob.glob(args.path)):
    paragraphs = read_dag(path)
    # print(path, len(paragraphs))

    if args.disamb:
        paragraphs_disamb = read_dag(args.disamb +'/'+os.path.basename(path))
        assert len(paragraphs) == len(paragraphs_disamb)
    else:
        paragraphs_disamb=[None] * len(paragraphs)

    for paragraph, paragraph_disamb in zip(paragraphs, paragraphs_disamb):
        years = paragraph['years']
        year_feature = years[:2]

        text = original_text(paragraph)
        reference_offsets=set()
        if paragraph_disamb:
            last_pos = 0
            for token in paragraph_disamb['tokens']:
                if is_disamb(token):
                    pos=text.index(token[SEGMENT], last_pos)
                    pos+=len(token[SEGMENT])
                    reference_offsets.add(pos)
                    last_pos=pos


        offsets = dag_offsets(paragraph, disamb_only=False, path=path)

        end_positions_tags = collections.defaultdict(set)
        for token in paragraph['tokens']:
            tags = set([interpretation[TAG] for interpretation in token['interpretations']])
            end_position = token[END_POSITION]
            end_positions_tags[end_position].update(tags)

        last_offset = 0
        for position, offset in sorted(offsets.items()):
            # print('K', position, offset, last_offset)
            segment = text[last_offset:offset]
            last_offset += len(segment)
            if segment[0] == ' ':
                space = '1'
                segment = segment[1:]
            else:
                space = '0'
            eot = '1' if offset in reference_offsets else '0'

            tags = end_positions_tags[position]
            poss = set([tag.split(':', 1)[0] for tag in tags])

            print('\t'.join([segment, space, '_'.join(sorted(tags)), '_'.join(sorted(poss)), year_feature, eot]))
            # print()

        print()
    print()
        
sys.exit()









def is_disamb(token):
    return any(['disamb' in interpretation['disamb'] for interpretation in token['interpretations']])


def is_manual_segmentation(token):
    return all(['manual' in interpretation['disamb'] for interpretation in token['interpretations']])


def gold_interpretation(token):
    for interpretation in token['interpretations']:
        if 'disamb' in interpretation['disamb']:
            return interpretation


def original_text(paragraph):
    strings = []
    for token in paragraph['tokens']:
        if any(['disamb' in interpretation['disamb'] for interpretation in token['interpretations']]):
            if token[SPACE_BEFORE]:
                strings.append(' ')
            strings.append(token[SEGMENT])
    return ''.join(strings)


parser = ArgumentParser(description='Train')
parser.add_argument('path', help='path to directory with data')
parser.add_argument('--only_disamb', action='store_true', help='print only disamb versions')
args = parser.parse_args()

for path in sorted(glob.glob(args.path)):
    paragraphs = read_dag(path)
    # print(path, len(paragraphs))
    for paragraph in paragraphs:
        years = paragraph['years']
        year_feature = years[:2]

        # find end positions with disamb*
        reference_end_positions = set([token[END_POSITION] for token in paragraph['tokens'] if is_disamb(token)])
        # print(reference_end_positions)
        nonmanual_end_positions = set(
            [token[END_POSITION] for token in paragraph['tokens'] if not is_manual_segmentation(token)])
        # print(nonmanual_end_positions)

        # dla nieznanych tagóœ dla tokenów ustawić pseudo token jako wyunik? i po,mijać go przy predykcji

        # 1. original text
        # 2. dla akzdego zakonczenia zapamietac tagi
        # 3. ciac i oznaczać space

        end_positions_tags = collections.defaultdict(set)
        for token in paragraph['tokens']:
            if is_manual_segmentation(token):
                continue
            tags = set([interpretation[TAG] for interpretation in token['interpretations'] if
                        'manual' not in interpretation['disamb']])
            end_position = token[END_POSITION]
            end_positions_tags[end_position].update(tags)

        text = original_text(paragraph)
        offsets = dag_offsets(paragraph)

        last_offset = 0
        for position, offset in sorted(offsets.items()):
            # print('K', position, offset, last_offset)
            segment = text[last_offset:offset]
            last_offset += len(segment)
            if segment[0] == ' ':
                space = '1'
                segment = segment[1:]
            else:
                space = '0'
            eot = '1' if position in reference_end_positions else '0'

            tags = end_positions_tags[position]
            poss = set([tag.split(':', 1)[0] for tag in tags])

            print(eot, segment, space, '_'.join(sorted(tags)), '_'.join(sorted(poss)), year_feature)
            # print()

        print()
    print()
