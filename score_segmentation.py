"""Merges output form Morfeusz woth reference data."""
from argparse import ArgumentParser
from typing import List, Tuple

import jsonlines

from ktagger import KText

parser = ArgumentParser(description='Score segmentation (ignore spaces)')
parser.add_argument('disamb_path', help='path to disamb JSONL (reference)')
parser.add_argument('pred_path', help='path to predictions (Flair output)')
parser.add_argument('tsv_path', help='path to TSV input data')

args = parser.parse_args()

reference_paragraphs = []
with jsonlines.open(args.disamb_path) as reader:
    for data in reader:
        ktext = KText.load(data)
        ktext.find_ambiguous_end_offsets()
        reference_paragraphs.append(ktext)

def input_paragraphs(path):
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
                segments.append((token, ambig))
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
                assert len(fields) in (3,4)
                token = fields[0]
                pred = int(fields[2])
                segments.append((token, pred))
    return segments


def get_reference_offsets(paragraph: KText):
    text = ''
    end_offsets = []
    last_offset = 0
    for token in paragraph.tokens:
        if token.has_disamb():
            form = token.form.replace(' ', '')
            last_offset += len(form)
            end_offsets.append(last_offset)
            text += form
    return end_offsets, text


def get_predicted_offsets(paragraph: List[Tuple[str, int]]):
    text = ''
    end_offsets = []
    last_offset = 0
    for token, decision in paragraph:
        last_offset += len(token)
        text += token
        if decision == 1:
            end_offsets.append(last_offset)
    return end_offsets, text

def get_input_unambig_offsets(paragraph: List[Tuple[str, int]]):
    text = ''
    unambig_offsets = []
    last_offset = 0
    for token, ambig in paragraph:
        last_offset += len(token)
        text += token
        if ambig == 0:
            unambig_offsets.append(last_offset)
    return unambig_offsets, text


def score(tp, fp, fn):
    print(f"TP: {tp} FP: {fp} FN: {fn}")
    precision = tp / (tp + fp) if tp+fp>0 else 0.0
    recall = tp / (tp + fn) if tp+fn>0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall>0 else 0.0
    print(f"Precision: {precision * 100:.4f} Recall: {recall * 100:.4f} F1: {f1 * 100:.4f}")


predicted_paragraphs = list(paragraphs(args.pred_path))
assert len(predicted_paragraphs) == len(reference_paragraphs)
input_paragraphs = list(input_paragraphs(args.tsv_path))
assert len(predicted_paragraphs) == len(input_paragraphs)

preds = {}
refs = {}
unambigs={}
for pred in predicted_paragraphs:
    pred_offsets, text = get_predicted_offsets(pred)
    preds[text] = set(pred_offsets)

for ref in reference_paragraphs:
    ref_offsets, text = get_reference_offsets(ref)
    refs[text] = set(ref_offsets)
    
for pred in input_paragraphs:
    unambig_offsets, text = get_input_unambig_offsets(pred)
    unambigs[text] = set(unambig_offsets)


print("\n".join(sorted(preds.keys() - refs.keys())))
print('---')
print("\n".join(sorted(refs.keys() - preds.keys())))


assert not (preds.keys() - refs.keys())
assert not (refs.keys() - preds.keys())
assert not (refs.keys() - unambigs.keys())
assert not (unambigs.keys() - refs.keys())

tp = fp = fn = 0
atp = afp = afn = 0
a = 0
for ref_text, ref_offsets in refs.items():
    pred_offsets = preds[ref_text]
    unambig_pred_offsets = unambigs[ref_text]

    tp += len(ref_offsets & pred_offsets)
    fn += len(ref_offsets - pred_offsets)
    fp += len(pred_offsets - ref_offsets)

    
    pred_offsets2 = pred_offsets - unambig_pred_offsets
    ambig_ref_offsets=ref_offsets-unambig_pred_offsets
    atp += len(ambig_ref_offsets & pred_offsets2)
    afn += len(ambig_ref_offsets - pred_offsets2)
    afp += len(pred_offsets2 - ambig_ref_offsets)

    # a += len(ambig_ref_offsets)
    # print(a)

print('ALL')
score(tp, fp, fn)
print('Ambig')
score(atp, afp, afn)
