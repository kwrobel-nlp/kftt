import sys
from argparse import ArgumentParser

import jsonlines


def paragraphs(path, tag_column=1, sep=' '):
    segments = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line == '':
                yield segments
                segments = []
            else:
                fields = line.split(sep)
                if len(fields) < 2:
                    print('ERROR', line, line.split(sep))
                token = fields[0]
                tag = fields[tag_column]
                segments.append((token, tag))
        if segments:
            yield segments
    return segments


if __name__ == '__main__':
    parser = ArgumentParser(description='')
    parser.add_argument('merged_path', help='path to merged JSONL')
    parser.add_argument('pred_path', help='path to TSV predictions (transformer output)')
    parser.add_argument('dir_path', help='path to dir DAG output')
    parser.add_argument('--tag_column', default=1, type=int, help='path to DAG output')
    parser.add_argument('--sep', default=' ', help='column separator in TSV')
    args = parser.parse_args()



    with jsonlines.open(args.merged_path) as reader:
        for data, segments in zip(reader, paragraphs(args.pred_path, tag_column=args.tag_column, sep=args.sep)):
            print(data['id'])
            id=data['id'].split('▁')[1]
            # continue
            with open(args.dir_path+'/'+id, 'a+') as writer:
            
                text = ''.join([token for token, pred in segments])
                # 1	2	wasze	wasz	adj:pl:acc:m:pos		disamb
                i=0
                for token, tag in segments:
                    nps='X'
                    writer.write('\t'.join([str(i),str(i+1),token,'X',tag,nps,'disamb']))
                    writer.write('\n')
                    i+=1
                writer.write('\n')