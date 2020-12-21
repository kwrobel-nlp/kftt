from score_segmentation import calculate, calculate2, calculate_sbd


def test_answer():
    disamb_path = 'data1/disamb.tsv'
    pred_path = 'data1/pred.tsv'
    ambig_path = 'data1/ambig.tsv'
    refs, preds, unambigs, input_refs, refs_sentence, preds_sentence, input_refs_sentence = calculate(disamb_path, pred_path, ambig_path)
    tp, fp, fn, precision, recall, f1, atp, afp, afn, aprecision, arecall, af1 = calculate2(refs, preds, unambigs)
    assert tp == 10
    assert fp == 1
    assert fn == 2
    assert atp == 3
    assert afp == 1
    assert afn == 2

    tp, fp, fn, precision, recall, f1, atp, afp, afn, aprecision, arecall, af1 = calculate2(input_refs, preds, unambigs)
    
def test_answer2():
    disamb_path = 'data2/disamb.tsv'
    pred_path = 'data2/pred.tsv'
    ambig_path = 'data2/ambig.tsv'
    refs, preds, unambigs, input_refs, refs_sentence, preds_sentence, input_refs_sentence = calculate(disamb_path, pred_path, ambig_path)
    tp, fp, fn, precision, recall, f1, atp, afp, afn, aprecision, arecall, af1 = calculate2(refs, preds, unambigs)
    assert tp == 10
    assert fp == 1
    assert fn == 3
    assert atp == 0
    assert afp == 1
    assert afn == 3

    tp, fp, fn, precision, recall, f1, atp, afp, afn, aprecision, arecall, af1 = calculate2(input_refs, preds, unambigs)

def test_answer3():
    disamb_path = 'data3/disamb.tsv'
    pred_path = 'data3/pred.tsv'
    ambig_path = 'data3/ambig.tsv'
    refs, preds, unambigs, input_refs, refs_sentence, preds_sentence, input_refs_sentence = calculate(disamb_path, pred_path, ambig_path)
    tp, fp, fn, precision, recall, f1, atp, afp, afn, aprecision, arecall, af1 = calculate2(refs, preds, unambigs)
    assert tp == 10
    assert fp == 3
    assert fn == 1
    assert atp == 0
    assert afp == 3
    assert afn == 1

    tp, fp, fn, precision, recall, f1, atp, afp, afn, aprecision, arecall, af1 = calculate2(input_refs, preds, unambigs)
    
def test_answer4():
    disamb_path = 'data4/disamb.tsv'
    pred_path = 'data4/pred.tsv'
    ambig_path = 'data4/ambig.tsv'
    refs, preds, unambigs, input_refs, refs_sentence, preds_sentence, input_refs_sentence = calculate(disamb_path, pred_path, ambig_path)
    print(refs_sentence)
    print(preds_sentence)
    tp, fp, fn, precision, recall, f1 = calculate_sbd(refs_sentence, preds_sentence,refs_sentence)
    assert tp == 1
    assert fp == 1
    assert fn == 1
    tp, fp, fn, precision, recall, f1 = calculate_sbd(refs_sentence, preds_sentence, refs_sentence, without_last=True)
    assert tp == 0
    assert fp == 1
    assert fn == 1