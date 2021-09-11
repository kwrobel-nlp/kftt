import hashlib
import sys
from argparse import ArgumentParser

import jsonlines
from tqdm import tqdm

from ktagger import KText
from utils2 import jsonlines_gzip_reader, jsonlines_gzip_writer

parser = ArgumentParser(
    description='Divides to CV folds using StratifiedKFold or GroupKFold. It doesnt split paragraphs in files.')
parser.add_argument('plain_path', help='path to plain JSONL')
parser.add_argument('-f', '--folds', type=int, default=10, help='number of folds')
parser.add_argument('--use_group_k_fold', action='store_true', help='use GroupKFold')
parser.add_argument('--random', action='store_true', help='random split')

args = parser.parse_args()

reference_paragraphs = {}
with jsonlines_gzip_reader(args.plain_path) as reader:
    for data in tqdm(reader):
        ktext = KText.load(data)
        reference_paragraphs[ktext.id] = ktext

ids=sorted(reference_paragraphs.keys())
import random
random.seed(0)
random.shuffle(ids)
print('HASH of ids order:', hashlib.sha256(str(ids).encode('utf-8')).hexdigest())


# print(ids)
groups=[id.rsplit('▁', 1)[0] for id in ids]
# print(groups)
if args.random:
    groups=[1]*len(ids)

# print(groups)

from sklearn.model_selection import StratifiedKFold, GroupKFold

if args.use_group_k_fold:
    kf = GroupKFold(n_splits=args.folds)
    splits=kf.split(ids, groups=groups)
else:
    kf = StratifiedKFold(n_splits=args.folds)
    splits = kf.split(ids, groups)
    
for i, (train, test) in tqdm(enumerate(splits)):
    #TODO sort by ids for testing, but not for tagging
    
    # print("%s %s" % (train, test))
    with jsonlines_gzip_writer(args.plain_path+"_"+str(i)+'_train') as writer:
        for j in train:
            writer.write(reference_paragraphs[ids[j]].save())
    with jsonlines_gzip_writer(args.plain_path+"_"+str(i)+'_test') as writer:
        for j in test:
            writer.write(reference_paragraphs[ids[j]].save())
    
