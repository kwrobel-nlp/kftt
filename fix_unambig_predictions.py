# Corrects unambig predictions
import sys
from argparse import ArgumentParser

def get_input_paragraphs(path):
    segments = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line == '':
                yield segments
                segments = []
            else:
                fields = line.split('\t')
                assert len(fields) == 7
                token = fields[0]
                ambig = int(fields[5])
                pred = int(fields[6])
                segments.append((token, ambig, pred))
        if segments:
            yield segments
    return segments


def paragraphs(path):
    segments = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line == '':
                yield segments
                segments = []
            else:
                fields = line.split(' ')
                # print(len(fields), fields)
                assert len(fields) in (3, 4)
                token = fields[0]
                pred = int(fields[2])
                segments.append((token, pred))
        if segments:
            yield segments
    return segments

if __name__ == '__main__':
    parser = ArgumentParser(description='Corrects unambig predictions')
    parser.add_argument('pred_path', help='path to predictions (Flair output)')
    parser.add_argument('tsv_path', help='path to TSV input data (with tokens marked as ambiguous)')
    args = parser.parse_args()

    ambig_paras={}
    for segments in get_input_paragraphs(args.tsv_path):
        text=''.join([token for token, ambig, pred in segments])
        ambig_paras[text]=segments
    
    for segments in paragraphs(args.pred_path):
        text = ''.join([token for token, pred in segments])
        
        ambig_para=ambig_paras[text]
        
        for (token, pred), (atoken, aambig, apred) in zip(segments, ambig_para):
            if aambig==0 and pred!=1:
                pred=1
                print('fix', file=sys.stderr)
            print(' '.join([token, 'X', str(pred), 'X']))
        print()
