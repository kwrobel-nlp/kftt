
NAME=$1

cut -f 1-6 data/$NAME-plain.segmentation.tsv.char > /tmp/a
cut -f 1-6 data/$NAME-merged.segmentation.tsv.char > /tmp/b
diff /tmp/a /tmp/b