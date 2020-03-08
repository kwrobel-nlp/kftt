# Create ensemble from transformers output

from collections import Counter


def most_frequent(l):
    occurence_count = Counter(l)
    # if l[0] !=occurence_count.most_common(1)[0][0]:
    #     print('XXX', l, occurence_count.most_common(1)[0][0])
    return occurence_count.most_common(1)[0][0]

paths=[
    'official_results/PolEval2020/XLMR-large/test_predictions.txt.ModelB+allF+CRF',
    'official_results/PolEval2020/XLMR-base/test_predictions.txt',
    'official_results/PolEval2020/SlavicBERT/test_predictions.txt',
    'official_results/PolEval2020/multiBERT/test_predictions.txt',
]


tokens=[]
tags=[]
for path in paths:
    for i,line in enumerate(open(path)):
        line=line.rstrip()
        if line=='':
            if len(tokens) <= i:
                tokens.append('')
                tags.append([])
            continue
        token, tag = line.split(' ')
        if len(tokens)<=i:
            tokens.append(token)
            tags.append([tag])
        else:
            tags[i].append(tag)
    # print(i)
# print(tokens[0], tags[0])
# print(len(tokens))

assert len(tokens)==len(tags)

for token, tag in zip(tokens, tags):
    if token=='':
        print()
    else:
        voted_tag=most_frequent(tag)
        print(token, voted_tag)