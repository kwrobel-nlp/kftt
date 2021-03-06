import logging
import random
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np
import torch

import flair.datasets
from flair.data import Corpus, Token, Label, Sentence
from flair.embeddings import (
    TokenEmbeddings,
    WordEmbeddings,
    StackedEmbeddings,
    FlairEmbeddings,
    CharacterEmbeddings,
    OneHotEmbeddings, BertEmbeddings)
from flair.training_utils import EvaluationMetric, add_file_handler
from flair.visual.training_curves import Plotter
from torch.utils.data.dataset import ConcatDataset

import tsv
from helpers import get_embeddings
from tokenization import FlairEmbeddingsEnd, FlairEmbeddingsBoth, FlairEmbeddingsOuter, FlairEmbeddingsStart

parser = ArgumentParser(description='Train')
parser.add_argument('data_folder', help='directory with corpus files')
parser.add_argument('train', help='train file name')
parser.add_argument('test', help='test file name')
parser.add_argument('--pretrained_model', help='path to pretrained model')

parser.add_argument('--dev', default=None, help='dev file name')

parser.add_argument('--corpora', nargs='+',
                    help='list of corpora for calculating label and feature spaces (for retraining)')

parser.add_argument('--output_folder', default='dot1', help='output folder for log and model')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--downsample', default=1.0, type=float, help='downsample ratio')
parser.add_argument('--downsample_train', action='store_true', help='downsample only train')

parser.add_argument('--hidden_size', default=256, type=int, help='size of embedding projection')
# parser.add_argument('--pretrained_model', default=None, help='path to pretrained model')
parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
parser.add_argument('--mini_batch_size', default=32, type=int, help='mini batch size')
parser.add_argument('--mini_batch_chunk_size', default=32, type=int, help='mini batch size chunk')
parser.add_argument('--max_epochs', default=300, type=int, help='max epochs')
parser.add_argument('--patience', default=5, type=int, help='patience')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
parser.add_argument('--rnn', default=1, type=int, help='number of RNN layers')
parser.add_argument('--tag_type', default='label',help='tag type to predict')
parser.add_argument('--embeddings_storage_mode', default='gpu', choices=['none', 'cpu', 'gpu'],
                    help='embeddings storage mode')
# parser.add_argument('--embeddings', nargs='+', help='list of embeddings, e.g. flair-pl-forward', required=True)
parser.add_argument('--monitor_train', action='store_true', help='evaluate train data after every epoch')
parser.add_argument('--tags', action='store_true', help='add maca tags as OneHot embeddings')
parser.add_argument('--poss', action='store_true', help='add maca poses as OneHot embeddings')
parser.add_argument('--space', action='store_true', help='add space as embeddings')
parser.add_argument('--year', action='store_true', help='add year as embeddings')
parser.add_argument('--crf', action='store_true', help='use CRF')
parser.add_argument('--use_amp', action='store_true', help='use AMP')
parser.add_argument('--amp_opt_level', default='O1', help='O1 or O2 for mixed precision')
parser.add_argument('--train_initial_hidden_state', action='store_true', help='train_initial_hidden_state')
parser.add_argument('--anneal_with_restarts', action='store_true', help='anneal_with_restarts')
parser.add_argument('--fine_tune_flair', action='store_true', help='fine_tune flair embeddings')
parser.add_argument('--w2v', action='store_true', help='w2v embeddings')
parser.add_argument('--mbert', action='store_true', help='mbert embeddings')
parser.add_argument('--fboth', action='store_true', help='FlairEmbeddingsBoth')
parser.add_argument('--start', action='store_true', help='FlairEmbeddingsStart')
parser.add_argument('--outer', action='store_true', help='FlairEmbeddingsOuter')

args = parser.parse_args()

log = logging.getLogger("args")
log.setLevel('INFO')
base_path = args.output_folder
if type(base_path) is str:
    base_path = Path(base_path)
log_handler = add_file_handler(log, base_path / "args.log")
log.addHandler(logging.StreamHandler())

log.info(str(args))

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# 1. get the corpus
columns = {0: 'text', 1: 'space_before', 2: 'tags', 3: 'poss', 4: 'year', 5: 'label'}
# sample 10% of train as dev

# 2. what tag do we want to predict?
tag_type = args.tag_type

# multicorpus for calculating feature spaces

corpora_paths=set()
if args.corpora is not None:
    corpora_paths.update(args.corpora)
corpora_paths.add(Path(args.data_folder) / args.train)

cs=[]
for c in corpora_paths:
    train = tsv.TSVDataset(
        Path(c),
        columns,
        tag_to_bioes=None,
        comment_symbol=None,
        in_memory=True,
        encoding="utf-8",
        document_separator_token=None
    )
    cs.append(train)
    print(train)
    
cd = ConcatDataset(cs)
cc=Corpus(cd, flair.datasets.SentenceDataset([]), flair.datasets.SentenceDataset([]))
tag_dictionary = cc.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)
# sys.exit()


corpus: Corpus = tsv.TSVCorpus(args.data_folder, columns,
                               train_file=args.train,
                               test_file=args.test,
                               dev_file=args.dev)

# set whitespace_after
for sentences in [corpus.train, corpus.dev, corpus.test]:
    for sentence in sentences:
        for i in range(1, len(sentence.tokens)):
            token: Token = sentence.tokens[i]
            token_before: Token = sentence.tokens[i - 1]
            if token.get_tag('space_before').value == '0':
                token_before.whitespace_after = False

# TODO: downsample - test 50% for trianig
# corpus.downsample(0.5, only_downsample_train=True)
if args.downsample<1.0:
    corpus = corpus.downsample(args.downsample, only_downsample_train=args.downsample_train)
print(corpus)



# 3. make the tag dictionary from the corpus
from flair.models import SequenceTagger
if args.pretrained_model:
    tagger: SequenceTagger = SequenceTagger.load(args.pretrained_model)
else:
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)
    
    # initialize embeddings
    if args.fboth:
        embedding_types: List[TokenEmbeddings] = [
            FlairEmbeddingsBoth('pl-forward', fine_tune=args.fine_tune_flair),  # TODO złe
            FlairEmbeddingsBoth('pl-backward', fine_tune=args.fine_tune_flair),
        ]
    elif args.start:
        embedding_types: List[TokenEmbeddings] = [
            FlairEmbeddingsStart('pl-forward', fine_tune=args.fine_tune_flair),  # TODO złe
            FlairEmbeddingsStart('pl-backward', fine_tune=args.fine_tune_flair),
        ]
    elif args.outer:
        embedding_types: List[TokenEmbeddings] = [
            FlairEmbeddingsOuter('pl-forward', fine_tune=args.fine_tune_flair),  # TODO złe
            FlairEmbeddingsOuter('pl-backward', fine_tune=args.fine_tune_flair),
        ]
    else:
        embedding_types: List[TokenEmbeddings] = [
            FlairEmbeddings('pl-forward', fine_tune=args.fine_tune_flair),  #TODO złe
            FlairEmbeddings('pl-backward', fine_tune=args.fine_tune_flair),
        ]
    if args.mbert:
        embedding_types.append(BertEmbeddings('/net/scratch/people/plgkwrobel/transformers/examples/pos-p2020-model_e20_multi_512/'))
    if args.w2v:
        embedding_types.append(WordEmbeddings('pl'))
        
    if args.tags:
        embedding_types.append(OneHotEmbeddings(corpus=cc, field='tags', embedding_length=20))
    if args.poss:
        embedding_types.append(OneHotEmbeddings(corpus=cc, field='poss', embedding_length=10))
    if args.space:
        embedding_types.append(OneHotEmbeddings(corpus=cc, field='space_before', embedding_length=2))
    if args.year:
        embedding_types.append(OneHotEmbeddings(corpus=cc, field='year', embedding_length=2))

        
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    
    # TODO class weights
    
    # initialize sequence tagger
    
    
    tagger: SequenceTagger = SequenceTagger(hidden_size=args.hidden_size,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=args.crf,
                                            rnn_layers=args.rnn,
                                            train_initial_hidden_state=args.train_initial_hidden_state,
                                            loss_weights={'0': 10.}
                                            )

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus, use_tensorboard=False)

# 7. start training
trainer.train(
    args.output_folder,
    learning_rate=args.learning_rate,
    mini_batch_size=args.mini_batch_size,
    mini_batch_chunk_size=args.mini_batch_chunk_size,
    max_epochs=args.max_epochs,
    min_learning_rate=1e-6,
    shuffle=True,
    anneal_factor=0.5,
    patience=args.patience,
    num_workers=args.num_workers,
    embeddings_storage_mode=args.embeddings_storage_mode,
    monitor_test=True,
    monitor_train=args.monitor_train,
    save_final_model=False,
    use_amp=args.use_amp,
    amp_opt_level=args.amp_opt_level,
    anneal_with_restarts=args.anneal_with_restarts
    )
