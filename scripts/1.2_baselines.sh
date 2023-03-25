NAME=$1

echo "ORACLE"
python3 jsonl_to_tsv_segmentation.py data/$NAME-merged.jsonl data/$NAME-merged.segmentation.tsv --eos
cut -f 1,2,7 data/$NAME-merged.segmentation.tsv | tr '\t' ' ' > data/$NAME-merged.oracle.tsv 
python3 score_segmentation.py data/$NAME-disamb.jsonl data/$NAME-merged.oracle.tsv data/$NAME-merged.segmentation.tsv

echo "SHORTEST"
python3 shortest_path.py data/$NAME-merged.jsonl > data/$NAME-merged.shortest.tsv
python3 score_segmentation.py data/$NAME-disamb.jsonl data/$NAME-merged.shortest.tsv data/$NAME-merged.segmentation.tsv

echo "LONGEST"
python3 shortest_path.py data/$NAME-merged.jsonl > data/$NAME-merged.longest.tsv --longest
python3 score_segmentation.py data/$NAME-disamb.jsonl data/$NAME-merged.longest.tsv data/$NAME-merged.segmentation.tsv
echo