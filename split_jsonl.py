import hashlib
import sys
from argparse import ArgumentParser

import jsonlines
from tqdm import tqdm

from ktagger import KText
from utils2 import jsonlines_gzip_reader, jsonlines_gzip_writer

parser = ArgumentParser(
    description='Split JSONL to two parts.')
parser.add_argument('plain_path', help='path to plain JSONL')
parser.add_argument('-r', '--ratio', type=float, default=0.9, help='ratio of training data')

args = parser.parse_args()

reference_paragraphs = {}
with jsonlines_gzip_reader(args.plain_path) as reader:
    for data in tqdm(reader):
        ktext = KText.load(data)
        reference_paragraphs[ktext.id] = ktext

ids = sorted(reference_paragraphs.keys())
import random

random.seed(0)
random.shuffle(ids)
print('HASH of ids order:', hashlib.sha256(str(ids).encode('utf-8')).hexdigest())

train_size: int = round(len(ids) * args.ratio)
train_ids = ids[:train_size]
dev_ids = ids[train_size:]

with jsonlines_gzip_writer(args.plain_path + '_train') as writer:
    for j in train_ids:
        writer.write(reference_paragraphs[j].save())
with jsonlines_gzip_writer(args.plain_path + '_dev') as writer:
    for j in dev_ids:
        writer.write(reference_paragraphs[j].save())
