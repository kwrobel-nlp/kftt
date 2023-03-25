mkdir data -p

for NAME in f19 korba_reczna nkjp1m; do
      echo $NAME
      python3 dag_to_jsonl.py "input_data/korba3/${NAME}/*/*ambig.dag" data/korba3-${NAME}-plain.jsonl korba3-${NAME} --korba
      python3 dag_to_jsonl.py "input_data/korba3/${NAME}/*/*disamb.dag" data/korba3-${NAME}-disamb.jsonl korba3-${NAME} --korba --only_disamb
done