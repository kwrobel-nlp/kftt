NAME=$1

python3 stats.py data/$NAME-plain.jsonl 
echo
python3 stats.py data/$NAME-disamb.jsonl 
echo
python3 stats.py data/$NAME-merged.jsonl 
echo