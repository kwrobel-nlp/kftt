
NAME=$1

python3 merge_plain_with_disamb.py data/$NAME-plain.jsonl data/$NAME-disamb.jsonl data/$NAME-merged.jsonl
python3 jsonl_to_tsv_segmentation_every_char.py data/$NAME-merged.jsonl data/$NAME-merged.segmentation.tsv.char --eos
python3 jsonl_to_tsv_segmentation_every_char.py data/$NAME-plain.jsonl data/$NAME-plain.segmentation.tsv.char --eos
