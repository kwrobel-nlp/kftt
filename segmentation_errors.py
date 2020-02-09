from argparse import ArgumentParser

from score_segmentation import calculate

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

        for a,b in sorted(ref_offsets - pred_offsets):
            print('FN:', a,b, ref_text[a:b])

        for a,b in sorted(pred_offsets - ref_offsets):
            print('FP:', a,b, ref_text[a:b])

        wrong_unambig =  unambig_pred_offsets - ref_offsets
        # print(wrong_unambig)
        # for a,b in sorted(wrong_unambig):
        #     print(a,b, ref_text[a:b])

        
        wrong_unambig =  ref_offsets - pred_offsets 
        # print(wrong_unambig)
        # for a,b in sorted(wrong_unambig):
        #     print(a,b, ref_text[a:b])
        print()


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