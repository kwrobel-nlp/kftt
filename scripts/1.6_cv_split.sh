NAME=$1
INDEX=$2

#python3 split_jsonl.py data/$NAME-plain.jsonl_${INDEX}_train
#python3 split_jsonl.py data/$NAME-disamb.jsonl_${INDEX}_train
#
#python3 merge_plain_with_disamb.py data/$NAME-plain.jsonl_${INDEX}_train_train data/$NAME-disamb.jsonl_${INDEX}_train_train data/$NAME-merged.jsonl_${INDEX}_train_train
#python3 merge_plain_with_disamb.py data/$NAME-plain.jsonl_${INDEX}_train_dev data/$NAME-disamb.jsonl_${INDEX}_train_dev data/$NAME-merged.jsonl_${INDEX}_train_dev
#python3 merge_plain_with_disamb.py data/$NAME-plain.jsonl_${INDEX}_test data/$NAME-disamb.jsonl_${INDEX}_test data/$NAME-merged.jsonl_${INDEX}_test

python3 jsonl_to_tsv_segmentation_every_char.py data/$NAME-merged.jsonl_${INDEX}_train_train data/$NAME-merged_${INDEX}_train_train.segmentation.tsv.char --eos
python3 jsonl_to_tsv_segmentation_every_char.py data/$NAME-merged.jsonl_${INDEX}_train_dev data/$NAME-merged_${INDEX}_train_dev.segmentation.tsv.char --eos
python3 jsonl_to_tsv_segmentation_every_char.py data/$NAME-merged.jsonl_${INDEX}_test data/$NAME-merged_${INDEX}_test.segmentation.tsv.char --eos  
    
