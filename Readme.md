# KFTT

Morphosyntactic tagger for Polish, the [winner](http://poleval.pl/results/) of PolEval 2020 Task 2: Morphosyntactic tagging of Middle, New and Modern Polish; successor of [KRNNT](https://github.com/kwrobel-nlp/krnnt).

|           Model          | Accuracy | Acc on known | Acc on ign | Acc on manual known |
|:------------------------:|:--------:|:------------:|:----------:|:-------------:|
| KFTT train+devel wo_morf |   95.63% |       95.95% |     81.91% |        67.30% |
| KFTT train+devel         |   95.73% |       96.07% |     81.02% |        67.81% |

KFTT train+devel accuracy on different parts of the test corpus:

|                        Corpus                      | Period | Accuracy | Acc on known | Acc on ign | Acc on manual |
|:--------------------------------------------------:|--------|:--------:|:------------:|:----------:|:-------------:|
| KorBa — a corpus of 17th and 18th century          | Middle | 94.35%   | 94.83%       | 79.43%     | 73.87%        |
| a corpus of 19th century                           | New    | 96.94%   | 97.15%       | 83.24%     | 78.39%        |
| 1M subcorpus of the National Corpus of Polish NKJP | Modern | 97.37%   | 97.48%       | 87.78%     | 84.07%        |

## PolEval 2020 results reproduction

Install PyTorch with a compatible version of CUDA with your drivers.
```commandline
pip install torch
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
wget https://github.com/kwrobel-nlp/kftt/releases/download/v0.1/ModelB+allF+CRF.pt
wget https://github.com/kwrobel-nlp/kftt/releases/download/v0.1/ModelB+CRF.pt
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

Download model:
```commandline
cd models
wget https://github.com/kwrobel-nlp/kftt/releases/download/v0.1/train+dev.zip
unzip train+dev.zip
cd ..
```

Tag:
```commandline
export BERT_MODEL=models/train+dev
time python3 run_ner_predict.py --data_dir ./test_data_temp/ \
--model_type xlmroberta \
--labels models/train+dev/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $BERT_MODEL \
--max_seq_length 512 \
--per_gpu_eval_batch_size 1 \
--do_predict
# Tagging: 16.61 seconds 
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