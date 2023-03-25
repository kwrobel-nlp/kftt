NAME=$1
CVN=${CVN:-10}

python3 do_cv.py data/$NAME-plain.jsonl -f $CVN
python3 do_cv.py data/$NAME-disamb.jsonl -f $CVN