
# PolEval 2020 results reproduction

Install PyTorch with a compatible version of CUDA with your drivers.
```commandline
pip install torch # torchvision
```
Install requirements:
```commandline
pip install -r requirements.txt
```
Download test data:
```commandline
NAME="test"

mkdir input_data
cd input_data
wget http://poleval.pl/task2/${NAME}-plain.tar.gz
tar -xf ${NAME}-plain.*
cd ..
```
Convert DAGs to JSONL format:
```commandline
mkdir data
python3 dag_to_jsonl.py "input_data/${NAME}-plain/*" data/${NAME}-plain.jsonl poleval2020-${NAME}
```
Prepare data for tokenization:
```commandline
python3 jsonl_to_tsv_segmentation_every_char.py data/${NAME}-plain.jsonl data/${NAME}-plain.segmentation.tsv.char
```
Download tokenization models:
```commandline
mkdir models
cd models
wget ModelB+allF+CRF.pt
wget ModelB+CRF.pt
cd ..
```
Choose tokenization model:
```commandline
MODEL_DIR="models/ModelB+allF+CRF.pt"
```
or (`wo_morf`):
```commandline
MODEL_DIR="models/ModelB+CRF.pt"
```
Tokenize:
```commandline
INPUT="data/${NAME}-plain.segmentation.tsv.char"
OUTPUT=${INPUT}.`basename "$MODEL_DIR"`
python3 predict_segmentation.py ${MODEL_DIR} $INPUT "$OUTPUT"
# Tokenization: 14.21 seconds
```

Prepare data for tagging:
```commandline
python3 test_data_tagging.py data/${NAME}-plain.jsonl $OUTPUT data/${NAME}-plain.segmentation.tsv.char ${OUTPUT}.tagging
echo ${OUTPUT}.tagging
# data/test-plain.segmentation.tsv.char.ModelB+allF+CRF.pt.tagging
mkdir test_data_temp
cp ${OUTPUT}.tagging test_data_temp/test.txt
```

Install transformers:
```commandline
pip3 install transformers==2.4.1 seqeval tensorboardX
```

Download model:
```commandline
cd models
wget 
unzip
cd ..
```

Tag:
```commandline
export MAX_LENGTH=512
export BERT_MODEL=models/pos-p2020-model_e20_xlmrl_512_run3-repro-train+dev+test
export OUTPUT_DIR=models/pos-p2020-model_e20_xlmrl_512_run3-repro-train+dev+test
export BATCH_SIZE=1
export NUM_EPOCHS=20
export SAVE_STEPS=574
export SEED=1
time python3 run_ner_predict.py --data_dir ./test_data_temp/ \
--model_type xlmroberta \
--labels models/pos-p2020-model_e20_xlmrl_512_run3-repro-train+dev+test/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_predict 
```

Copy results:
```commandline
cp ${OUTPUT_DIR}/test_predictions_long.txt ${OUTPUT}.tagged
```

Join results and split to separate files:
```commandline
mkdir predictions
python3 split_to_separate_files.py data/${NAME}-plain.jsonl ${OUTPUT}.tagged predictions --sep $'\t'
```
Evaluate:
```commandline
python3 poleval-eval.py predictions input_data/${NAME}-disamb 2>/dev/null
```
Results:
```
Poleval 2020 Task 2

Evaluating files in: predictions
against 252 gold standard files in: input_data/test-disamb


Accuracy (Your score!): 0.9572980397053314

Tokens total:           40045
Correct tokens:         38335
Unknown tokens:         901 (2.25%)
Correct unknown:        730
Accuracy on unknown:    0.8102108768035516
Known tokens:           39144 (97.75%)
Accuracy on known:      0.960683629675046
Manual tokens:          1482 (known 581 + ign 901)
Correct manual:         1124
Accuracy on manual:     0.7584345479082322
Accuracy manual known:  0.6781411359724613
Accuracy manual ign:    0.8102108768035516
```