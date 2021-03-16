"""Merges output form Morfeusz woth reference data."""
import collections
import sys
from argparse import ArgumentParser

import jsonlines

from ktagger import KText, KToken


def paragraphs(path, input_path):
    segments = []
    with open(path) as f, open(input_path) as f2:
        for line,line2 in zip(f,f2):
            line = line.rstrip()
            line2 = line2.rstrip()
            if line == '':
                yield segments
                segments = []
            else:
                fields = line.split(' ')
                # print(len(fields), fields)
                assert len(fields) in (4, )
                token = fields[0]
                pred = int(fields[2])

                fields2 = line2.split('\t')
                space_before=int(fields2[1])
                segments.append((token, pred, space_before))
        if segments:
            yield segments
    return segments

def segments_to_tokens(segments):
    tokens = []
    token = ''
    for char, pred, space_before in segments:
        if token != '':
            if space_before==1:
                token+=' '
        
        token += char
        
        #TODO brakuje info o spacjach, jeśli klasyfikator nie podzielił przy spacji: "dowolny'czas'" 
        # a powinno być "dowolny 'czas'" - prawdopodobnie fix ambig to załatwia
        if pred >= 1:
            tokens.append(token)
            token = ''
    assert token==''
    return tokens

parser = ArgumentParser(description='Reads segmentation per char predictions (order of texts must be the same).')
parser.add_argument('merged_path', help='path to merged JSONL')
parser.add_argument('pred_path', help='path to segmentation predictions TSV')
parser.add_argument('input_path', help='path to TSV with info about spaces')
parser.add_argument('output_path', help='path to output TSV')
# parser.add_argument('--only_disamb', action='store_true', help='save only disamb versions of tokens and interpretations')
args = parser.parse_args()

#get all tokens from char preds

# for segments in paragraphs(args.pred_path):
#     tokens=segments_to_tokens(segments)
#     print(tokens)
#     # break

with jsonlines.open(args.merged_path) as reader, open(args.output_path, 'w') as writer:
    for data, segments in zip(reader, paragraphs(args.pred_path, args.input_path)):
        segmentation_tokens = segments_to_tokens(segments)
        # print(segmentation_tokens)
        
            
        ktext = KText.load(data)

        ktext.tokens = sorted(ktext.tokens, key=lambda t: (t.start_position, t.end_position))
        
        
        ktext.fix_offsets3()
        #find offsets of each token in text without spaces?

        plain_tokens={}
        for token in ktext.tokens:
            if token.manual:
                continue
            # print(token.form, token.start_offset2, token.end_offset2, token.space_before)
            plain_tokens[(token.form, token.start_offset2, token.end_offset2)]=token
                
        predicted_tokens=[]
        start_offset = 0
        for token in segmentation_tokens:
            end_offset=start_offset+ len(token)
            # print(token, start_offset, end_offset)
            if (token, start_offset, end_offset) in plain_tokens:
                # print('OK')
                plain_token=plain_tokens[(token, start_offset, end_offset)]
                predicted_tokens.append(plain_token)
            else:
                # print('BAD')
                #craete new token
                plain_token=KToken(form=token, space_before=None, start_offset=None, end_offset=None)
                predicted_tokens.append(plain_token)
            
            start_offset = end_offset
        #TODO: fix offsets and space_before: fixoffsets2?
        ktext.tokens=predicted_tokens
        ktext.fix_offsets(ktext.text)
        
        # fix space_before
        for token in ktext.tokens:
            if token.space_before is None:
                token.space_before = ktext.text[token.start_offset-1]==' '
                
        for token in ktext.tokens:
            # print(token.form, token.start_offset, token.end_offset, token.space_before)
        # break
           
                
            tags = set([interpretation.tag for interpretation in token.interpretations])
            if not tags:
                tags.add('ign')
            poss = set([tag.split(':', 1)[0] for tag in tags])
            space = '1' if token.space_before else '0'

            writer.write("\t".join(
                [token.form, space, '_'.join(sorted(tags)), '_'.join(sorted(poss)), str(ktext.year)]))
            writer.write("\n")
        writer.write("\n")
