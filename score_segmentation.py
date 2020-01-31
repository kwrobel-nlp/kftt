"""Merges output form Morfeusz woth reference data."""
from argparse import ArgumentParser
from typing import List, Tuple

import jsonlines

from ktagger import KText

parser = ArgumentParser(description='Score segmentation (ignore spaces)')
parser.add_argument('disamb_path', help='path to disamb JSONL (reference)')
parser.add_argument('pred_path', help='path to predictions')

args = parser.parse_args()

reference_paragraphs = []
with jsonlines.open(args.disamb_path) as reader:
    for data in reader:
        ktext = KText.load(data)
        reference_paragraphs.append(ktext)

def paragraphs(path):
    segments=[]
    with open(path) as f:
        for line in f:
            line=line.rstrip()
            if line=='':
                yield segments
                segments=[]
            else:
                fields = line.split(' ')
                assert len(fields)==4
                token=fields[0]
                pred=int(fields[2])
                segments.append((token, pred))

def get_reference_offsets(paragraph: KText):
    text=''
    end_offsets=[]
    last_offset=0
    for token in paragraph.tokens:
        if token.has_disamb():
            form=token.form.replace(' ','')
            last_offset+=len(form)
            end_offsets.append(last_offset)
            text+=form
    return end_offsets, text

def get_predicted_offsets(paragraph: List[Tuple[str, int]]):
    text = ''
    end_offsets = []
    last_offset = 0
    for token, decision in paragraph:
        last_offset += len(token)
        text += token
        if decision==1:
            end_offsets.append(last_offset)
    return end_offsets, text

predicted_paragraphs = list(paragraphs(args.pred_path))
assert len(predicted_paragraphs)==len(reference_paragraphs)

preds={}
refs={}

for pred in predicted_paragraphs:
    pred_offsets, text = get_predicted_offsets(pred)
    preds[text]=set(pred_offsets)

for ref in reference_paragraphs:
    ref_offsets, text = get_reference_offsets(ref)
    refs[text]=set(ref_offsets)


assert not (preds.keys()-refs.keys())
assert not (refs.keys()-preds.keys())

tp=fp=fn=0
for ref_text, ref_offsets in refs.items():
    pred_offsets=preds[ref_text]
    tp+=len(ref_offsets&pred_offsets)
    fn+=len(ref_offsets-pred_offsets)
    fp+=len(pred_offsets-ref_offsets)
    
print(f"TP: {tp} FP: {fp} FN: {fn}")
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f1=2*precision*recall/(precision+recall)
print(f"Precision: {precision*100:.4f} Recall: {recall*100:.4f} F1: {f1*100:.4f}")

