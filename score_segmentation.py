"""Merges output form Morfeusz woth reference data."""
from argparse import ArgumentParser
from typing import List, Tuple

import jsonlines

from ktagger import KText




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


def get_reference_offsets(paragraph: KText):
    text = ''
    end_offsets = []
    last_offset = 0
    for token in paragraph.tokens:
        if token.has_disamb() or len(token.interpretations)==0: # in UGC missing interpretations
            form = token.form.replace(' ', '')
            end_offsets.append((last_offset, last_offset + len(form)))
            last_offset += len(form)
            text += form
    return end_offsets, text


def get_predicted_offsets(paragraph: List[Tuple[str, int]]):
    text = ''
    end_offsets = []
    last_offset = 0
    last_start_offset = 0
    for token, decision in paragraph:
        text += token
        if decision == 1:
            end_offsets.append((last_start_offset, last_offset + len(token)))
            last_start_offset = last_offset + len(token)
        last_offset += len(token)
    return end_offsets, text


def get_input_offsets(paragraph: List[Tuple[str, int, int]]):
    text = ''
    end_offsets = []
    last_offset = 0
    last_start_offset = 0
    for token, ambig, decision in paragraph:
        token = token.replace(' ', '')
        text += token
        if decision == 1:
            end_offsets.append((last_start_offset, last_offset + len(token)))
            last_start_offset = last_offset + len(token)
        last_offset += len(token)
    return end_offsets, text


def get_input_unambig_offsets(paragraph: List[Tuple[str, int, int]]):
    text = ''
    unambig_offsets = []
    last_offset = 0
    prev_ambig = False
    for token, ambig, decision in paragraph:
        text += token
        if ambig == 0:
            if not prev_ambig:
                unambig_offsets.append((last_offset, last_offset + len(token)))
            prev_ambig = False
        else:
            prev_ambig = True
        last_offset += len(token)
    return unambig_offsets, text


def score(tp, fp, fn):
    print(f"TP: {tp} FP: {fp} FN: {fn}")
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    print(f"Precision: {precision * 100:.4f} Recall: {recall * 100:.4f} F1: {f1 * 100:.4f}")
    return precision, recall, f1


def calculate(disamb_path, pred_path, ambig_path):
    if 'jsonl' in disamb_path:
        reference_paragraphs = []
        with jsonlines.open(disamb_path) as reader:
            for data in reader:
                ktext = KText.load(data)
                # ktext.find_ambiguous_end_offsets()
                reference_paragraphs.append(ktext)

        refs = {}
        for ref in reference_paragraphs:
            ref_offsets, text = get_reference_offsets(ref)
            refs[text] = set(ref_offsets)
    elif 'tsv' in disamb_path:
        reference_paragraphs = list(get_input_paragraphs(disamb_path))
        refs = {}
        for pred in reference_paragraphs:
            input_offsets, text = get_input_offsets(pred)
            refs[text] = set(input_offsets)

    predicted_paragraphs = list(paragraphs(pred_path))
    assert len(predicted_paragraphs) == len(reference_paragraphs)
    input_paragraphs = list(get_input_paragraphs(ambig_path))
    assert len(predicted_paragraphs) == len(input_paragraphs)

    preds = {}

    input_refs = {}
    unambigs = {}
    for pred in predicted_paragraphs:
        pred_offsets, text = get_predicted_offsets(pred)
        preds[text] = set(pred_offsets)

    for pred in input_paragraphs:
        unambig_offsets, text = get_input_unambig_offsets(pred)
        unambigs[text] = set(unambig_offsets)

    for pred in input_paragraphs:
        input_offsets, text = get_input_offsets(pred)
        input_refs[text] = set(input_offsets)

    print("\n".join(sorted(preds.keys() - refs.keys())))
    print('---')
    print("\n".join(sorted(refs.keys() - preds.keys())))

    assert not (preds.keys() - refs.keys())
    assert not (refs.keys() - preds.keys())
    assert not (refs.keys() - unambigs.keys())
    assert not (unambigs.keys() - refs.keys())

    return refs, preds, unambigs, input_refs


def calculate2(refs, preds, unambigs):
    tp = fp = fn = 0
    atp = afp = afn = 0
    a = 0
    for ref_text, ref_offsets in refs.items():
        pred_offsets = preds[ref_text]
        unambig_pred_offsets = unambigs[ref_text]
        
        tp += len(ref_offsets & pred_offsets)
        fn += len(ref_offsets - pred_offsets)
        fp += len(pred_offsets - ref_offsets)

        # print(unambig_pred_offsets)

        pred_offsets2 = pred_offsets - unambig_pred_offsets
        ambig_ref_offsets = ref_offsets - unambig_pred_offsets
        atp += len(ambig_ref_offsets & pred_offsets2)
        afn += len(ambig_ref_offsets - pred_offsets2)
        afp += len(pred_offsets2 - ambig_ref_offsets)

        a += len(ambig_ref_offsets)
    print(a)

    print('ALL')
    precision, recall, f1 = score(tp, fp, fn)

    print('Ambig')
    aprecision, arecall, af1 = score(atp, afp, afn)

    return tp, fp, fn,precision, recall, f1, atp, afp, afn, aprecision, arecall, af1


if __name__ == '__main__':
    parser = ArgumentParser(description='Score segmentation (ignore spaces)')
    parser.add_argument('disamb_path', help='path to disamb JSONL or TSV (reference)')
    parser.add_argument('pred_path', help='path to predictions (Flair output)')
    parser.add_argument('tsv_path', help='path to TSV input data (with tokens marked as ambiguous)')

    args = parser.parse_args()
    
    refs, preds, unambigs, input_refs = calculate(args.disamb_path, args.pred_path, args.tsv_path)
    calculate2(refs, preds, unambigs)

    print('Against training')
    calculate2(input_refs, preds, unambigs)
