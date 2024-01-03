# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """


import argparse
import glob
import logging
import os
import random
import sys

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_file, InputExample


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

#ALL_MODELS = sum(
    #(
        #tuple(conf.pretrained_config_archive_map.keys())
        #for conf in (BertConfig, RobertaConfig, DistilBertConfig, CamembertConfig, XLMRobertaConfig)
    #),
    #(),
#)
ALL_MODELS=[]
MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def expand(word_tokens, max_length, start, end):
    counter = 0
    for i in range(start, end):
        counter += word_tokens[i]
    # print(counter)
    added = True
    while added:
        added = False
        if start - 1 >= 0 and counter + word_tokens[start - 1] <= max_length:
            start -= 1
            counter += word_tokens[start]
            added = True
        if end < len(word_tokens) and counter + word_tokens[end] <= max_length:
            counter += word_tokens[end]
            end += 1
            added = True
    # print(start, end, counter)
    return start, end


def windows(word_tokens, max_length, min_context):
    
    results = []
    content = max_length - 2 * min_context
    start_content = 0
    end = min_context + content

    start_counter = 0

    counter = 0
    i = 0
    while i < len(word_tokens):
        if counter + word_tokens[i] > end or i == len(word_tokens) - 1:

            end_content = i
            start_all, end_all = expand(word_tokens, max_length, start_content, i)
            if end_all == len(word_tokens):
                end_content = end_all
                i = end_content - 1
            # print(start_all, start_content, end_content, end_all)
            results.append((start_all, start_content, end_content, end_all))
            start_content = i
            start_counter = counter
            end += content

        counter += word_tokens[i]
        i += 1
    return results


# windows(word_tokens, max_length, min_context)





def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    #eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()


    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file

    logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = read_examples_from_file(args.data_dir, mode)
    # print(len(examples))
    # print(examples[0])  # list of words in one document (sentence)

    out_label_listX=[]
    preds_listX=[]

    for example in tqdm(examples, desc="Evaluating"):
        max_length = 500
        min_context = 128
        l = len(example.words) #TODO number of segments for each word? word_tokens = tokenizer.tokenize(word)
        word_tokens_lengths=[len(tokenizer.tokenize(word)) for word in example.words]

        ws=windows(word_tokens_lengths, max_length, min_context)
        # print(ws)
        
        text_examples=[]
        for start_all, start_content, end_content, end_all in ws:
            ex=InputExample(guid=example.guid, words=example.words[start_all:end_all], labels=example.labels[start_all:end_all])
            text_examples.append(ex)

        # przed tym trzeba podzielić
        features = convert_examples_to_features(
            text_examples,
            labels,
            512, #args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id,
        )
    
        if args.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    
        eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    







    
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    

        # Eval!

        logger.info("  Num examples = %d", len(eval_dataset))
        
        # batch = next(iter(eval_dataloader))
        a = []
        b = []
        for batch, (start_all, start_content, end_content, end_all) in tqdm(zip(eval_dataloader, ws), desc="Evaluating"):
            preds = None
            out_label_ids = None
            batch = tuple(t.to(args.device) for t in batch)
    
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
    
                if args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
    
                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        
            
            preds = np.argmax(preds, axis=2)
        
            label_map = {i: label for i, label in enumerate(labels)}
        
            out_label_list = [[] for _ in range(out_label_ids.shape[0])]
            preds_list = [[] for _ in range(out_label_ids.shape[0])]
        
            for i in range(out_label_ids.shape[0]):
                for j in range(out_label_ids.shape[1]):
                    if out_label_ids[i, j] != pad_token_label_id:
                        out_label_list[i].append(label_map[out_label_ids[i][j]])
                        preds_list[i].append(label_map[preds[i][j]])
    
            #join
            
            for i in range(len(out_label_list)):
                a.extend(out_label_list[i][start_content-start_all:end_content-start_all])
                b.extend(preds_list[i][start_content-start_all:end_content-start_all])
    
        out_label_listX.append(a)
        preds_listX.append(b)
        # results = {
        #     "loss": eval_loss,
        #     "precision": precision_score(out_label_list, preds_list),
        #     "recall": recall_score(out_label_list, preds_list),
        #     "f1": f1_score(out_label_list, preds_list),
        # }

    eval_loss = eval_loss / nb_eval_steps

    try:
        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_listX, preds_listX),
            "recall": recall_score(out_label_listX, preds_listX),
            "f1": f1_score(out_label_listX, preds_listX),
            "cr": classification_report(out_label_listX, preds_listX),
        }
        
        logger.info("***** Eval results %s *****", prefix)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        
        return results, preds_listX
    except IndexError: #no output labels in file
        return {}, preds_listX


class Tagger():
    def __init__(self, labels, model_type, model_name_or_path, no_cuda=False, cache_dir=None, per_gpu_eval_batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count()

        # labels = get_labels(args.labels)
        self.labels=labels
        num_labels = len(labels)
        self.per_gpu_eval_batch_size=per_gpu_eval_batch_size

        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        
        model_type = model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        # config = config_class.from_pretrained(
        #     model_name_or_path,
        #     num_labels=num_labels,
        #     cache_dir=cache_dir if cache_dir else None,
        # )
        # tokenizer = tokenizer_class.from_pretrained(
        #     args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        #     do_lower_case=args.do_lower_case,
        #     cache_dir=args.cache_dir if args.cache_dir else None,
        # )
        # self.model = model_class.from_pretrained(
        #     model_name_or_path,
        #     from_tf=bool(".ckpt" in model_name_or_path),
        #     config=config,
        #     cache_dir=cache_dir if cache_dir else None,
        # )

        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=False)
        self.model = model_class.from_pretrained(model_name_or_path)
        self.model.to(self.device)

        self.eval_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)
    
        # multi-gpu evaluate
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.eval()

    def predict(self, examples):
        # import time
        # start_time = time.time()
        result, predictions, probs = self.evaluate(examples, mode="test")
        # print("--- %s seconds ---" % (time.time() - start_time))
        return result, predictions, probs
        
    def evaluate(self, examples, mode):
        # eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

        
            
        # Eval!
        logger.info("***** Running evaluation *****")
        logger.info("  Batch size = %d", self.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        # preds = None
        # out_label_ids = None
        

        # Load data features from cache or dataset file

        logger.info("Creating features from dataset file at")
        
        # print('len(examples)', len(examples))
        # print(examples[0])  # list of words in one document (sentence)

        out_label_listX = []
        preds_listX = []
        preds_prob_listX = []

        # for example in tqdm(examples, desc="Evaluating"):
        for example in examples:
            max_length = 500
            min_context = 128
            l = len(example.words)  # TODO number of segments for each word? word_tokens = tokenizer.tokenize(word)
            word_tokens_lengths = [len(self.tokenizer.tokenize(word)) for word in example.words]

            ws = windows(word_tokens_lengths, max_length, min_context)
            print('ws', ws)

            text_examples = []
            for start_all, start_content, end_content, end_all in ws:
                ex = InputExample(guid=example.guid, words=example.words[start_all:end_all],
                                  labels=example.labels[start_all:end_all])
                text_examples.append(ex)

            # print('len(text_examples)', len(text_examples))
            # przed tym trzeba podzielić
            features = convert_examples_to_features(
                text_examples,
                self.labels,
                512,  # args.max_seq_length,
                self.tokenizer,
                cls_token_at_end=False,
                # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                cls_token_segment_id=0,
                sep_token=self.tokenizer.sep_token,
                sep_token_extra=False,
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=False,
                # pad on the left for xlnet
                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                pad_token_segment_id=0,
                pad_token_label_id=self.pad_token_label_id,
            )

            # print('features', features)

            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

            eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.eval_batch_size)

            # Eval!

            logger.info("  Num examples = %d", len(eval_dataset))

            # batch = next(iter(eval_dataloader))
            a = []
            b = []
            c=[]
            for batch, (start_all, start_content, end_content, end_all) in zip(eval_dataloader, ws):
                preds = None
                out_label_ids = None
                batch = tuple(t.to(self.device) for t in batch)

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    inputs["token_type_ids"] = None
                    outputs = self.model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    if self.n_gpu > 1:
                        tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

                    eval_loss += tmp_eval_loss.item()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

                pred_probs=torch.max(torch.softmax(torch.tensor(preds), 2), 2).values #np.max(preds, axis=2)
                # print('pred_probs', pred_probs)
                preds = np.argmax(preds, axis=2)

                label_map = {i: label for i, label in enumerate(self.labels)}

                out_label_list = [[] for _ in range(out_label_ids.shape[0])]
                preds_list = [[] for _ in range(out_label_ids.shape[0])]
                preds_prob_list = [[] for _ in range(out_label_ids.shape[0])]

                # print('out_label_ids.shape', out_label_ids.shape)
                for i in range(out_label_ids.shape[0]):
                    for j in range(out_label_ids.shape[1]):
                        if out_label_ids[i, j] != self.pad_token_label_id:
                            out_label_list[i].append(label_map[out_label_ids[i][j]])
                            preds_list[i].append(label_map[preds[i][j]])
                            preds_prob_list[i].append(pred_probs[i][j])

                # join

                for i in range(len(out_label_list)):
                    a.extend(out_label_list[i][start_content - start_all:end_content - start_all])
                    b.extend(preds_list[i][start_content - start_all:end_content - start_all])
                    c.extend(preds_prob_list[i][start_content - start_all:end_content - start_all])

            # print('len(a), len(b)', len(a), len(b))
            out_label_listX.append(a)
            preds_listX.append(b)
            preds_prob_listX.append(c)
            # results = {
            #     "loss": eval_loss,
            #     "precision": precision_score(out_label_list, preds_list),
            #     "recall": recall_score(out_label_list, preds_list),
            #     "f1": f1_score(out_label_list, preds_list),
            # }

        eval_loss = eval_loss / nb_eval_steps

        try:
            results = {
                "loss": eval_loss,
                "precision": precision_score(out_label_listX, preds_listX),
                "recall": recall_score(out_label_listX, preds_listX),
                "f1": f1_score(out_label_listX, preds_listX),
                "cr": classification_report(out_label_listX, preds_listX),
            }

            logger.info("***** Eval results *****")
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))

            return results, preds_listX, preds_prob_listX
        except IndexError:  # no output labels in file
            return {}, preds_listX, preds_prob_listX


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    labels = get_labels(args.labels)
    examples = read_examples_from_file(args.data_dir, 'test')
    tagger=Tagger(labels, args.model_type, args.model_name_or_path, no_cuda=args.no_cuda, cache_dir=args.cache_dir, per_gpu_eval_batch_size=args.per_gpu_eval_batch_size)
    result, predictions, probs = tagger.predict(examples)

    # Save results
    output_test_results_file = os.path.join(args.output_dir, "test_results_long.txt")
    with open(output_test_results_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("{} = {}\n".format(key, str(result[key])))
    # Save predictions
    output_test_predictions_file = os.path.join(args.output_dir, "test_predictions_long.txt")
    with open(output_test_predictions_file, "w") as writer:
        with open(os.path.join(args.data_dir, "test.txt"), "r") as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not predictions[example_id]:
                        example_id += 1
                elif predictions[example_id]:
                    output_line = line.split("\t")[0] + "\t" + predictions[example_id].pop(0) + "\n"
                    writer.write(output_line)
                else:
                    logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split("\t")[0])


if __name__ == "__main__":
    main()
