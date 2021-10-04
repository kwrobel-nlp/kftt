import collections
import json
import os
from argparse import ArgumentParser

from flair.models import SequenceTagger

from dag_to_jsonl import convert_to_ktagger
from jsonl_to_tsv_segmentation_every_char2 import FlairConverter
from ktagger import KText, KToken, KInterpretation
from run_ner_predict2 import Tagger
from test_data_tagging import merge, merge2
from utils_ner import get_labels, InputExample

class Lemmatizer():
    def __init__(self, path=None):
        if path:
            self.stats_form_tag_lemma, self.stats_lemma=json.load(open(path))
        else:
            self.stats_form_tag_lemma={}
            self.stats_lemma={}
    
    def stat_lemmatize(self, form, tag):
        # print('LEMMATIZE:', form, tag)
        if form in self.stats_form_tag_lemma and tag in self.stats_form_tag_lemma[form]:
            lemma=sorted(self.stats_form_tag_lemma[form][tag].items(), key=lambda x: x[1], reverse=True)[0][0]
            # print('FOUND LEMMA:', lemma)
            return lemma
    
    def stat_lemmatize_fuzzy(self, form, tag):
        # print('LEMMATIZE:', form, tag)
        lemmas=collections.defaultdict(int)
        if form in self.stats_form_tag_lemma:
            for tag, lemma_count in self.stats_form_tag_lemma[form].items():
                for lemma, count in lemma_count.items():
                    lemmas[lemma]+=count
            
            lemma=sorted(lemmas.items(), key=lambda x: x[1], reverse=True)[0][0]
            print('FOUND FUZZY LEMMA:', lemma)
            return lemma
    
    def lemmatize(self, token: KToken):
        form = token.form
        predicted_tag = token.predicted

        stat_lemma = self.stat_lemmatize(form, predicted_tag)
        if stat_lemma:
            return stat_lemma

        if token.interpretations:
            # 1. sprawdz czy sa interpretacje z takim samym tagiem
            # print(token.form, token.interpretations[0].tag)

            matched_lemmas = []
            for interp in token.interpretations:
                # print(interp.lemma, interp.tag)
                if predicted_tag == interp.tag:
                    matched_lemmas.append(interp.lemma)
                    # print('-', interp.lemma)

            matched_lemmas = list(set(matched_lemmas))
            if len(matched_lemmas) == 1:
                stat_lemma=self.stat_lemmatize(form, predicted_tag)
                if stat_lemma and stat_lemma !=matched_lemmas[0]:
                    print('LEMMATIZER: 1', stat_lemma, matched_lemmas[0])
                return matched_lemmas[0]
            elif len(matched_lemmas) > 1:
                print('More than 1 matched lemma:', form, predicted_tag, matched_lemmas)
                stat_lemma = self.stat_lemmatize(form, predicted_tag)
                if stat_lemma:
                    print('LEMMATIZER: 2', stat_lemma, matched_lemmas[0])
                    return stat_lemma
                return matched_lemmas[0]  # TODO może zwracać obie?

            # 2. lemat dla tagu o największej części wspólnej od początku, przy czym cz. mowy musi być taka sama?
            predicted_pos = predicted_tag.split(':')[0]
            prefixes = collections.defaultdict(list)
            with_other_pos = []
            for interp in token.interpretations:
                if interp.tag.startswith(predicted_pos):
                    # print(form, predicted_tag, interp.lemma, interp.tag)
                    prefix = os.path.commonprefix([predicted_tag, interp.tag])
                    prefixes[len(prefix)].append(interp.lemma)
                else:
                    # print('Different POS:', form, predicted_tag, interp.lemma, interp.tag)
                    with_other_pos.append(interp)
            # print(prefixes)
            if prefixes:
                most_common = list(set(max(prefixes.items())[1]))
                if len(most_common) == 1:
                    return most_common[0]
                else:
                    print('More than 1 lemma with most common tag:', form, predicted_tag, most_common)
                    print([(interp.tag, interp.lemma) for interp in token.interpretations])
                    return most_common[0]  # TODO

            if not prefixes and with_other_pos:  # brak interpretacji z taką samą cz. mowy ale istnieją z inną
                non_ign_lemmas = []
                for interp in with_other_pos:
                    if interp.tag != 'ign':
                        print('Only with different POS:', form, predicted_tag, interp.lemma, interp.tag)  # TODO
                        non_ign_lemmas.append(interp.lemma)
                if non_ign_lemmas:
                    return non_ign_lemmas[0]  # TODO

        else:
            print('No plain_token:', form, predicted_tag)
            #find lemma from stats with different tag
            stat_lemma=self.stat_lemmatize_fuzzy(form, predicted_tag)
            if stat_lemma:
                return stat_lemma

        return form

class KFTT():
    def __init__(self, tokenizer_path, tagger_path, model_type, no_cuda, cache_dir, per_gpu_eval_batch_size, lemmatizer_path,
                 tokenizer_batch_size=32):
        self.flair_converter = FlairConverter()
        self.tagger: SequenceTagger = SequenceTagger.load(tokenizer_path)
        self.tokenizer_batch_size = tokenizer_batch_size

        labels = get_labels(tagger_path + '/labels.txt')

        self.tagger2 = Tagger(labels, model_type, tagger_path, no_cuda=no_cuda,
                              cache_dir=cache_dir, per_gpu_eval_batch_size=per_gpu_eval_batch_size)
        
        self.lemmatizer=Lemmatizer(lemmatizer_path)

    def tag(self, text: KText) -> KText:
        # 1. convert to flair.Sentences every char
        sentence = self.flair_converter.convert_to_sentence(text)
        self.tagger.predict(sentence, mini_batch_size=self.tokenizer_batch_size)
        # 2. predictions back to ktext - chyba łątwiej utworzyć nowe, ale trzeba przenieść info o morfologii
        segments = []
        for token in sentence.tokens:
            char = token.text
            pred = int(token.get_tag('label').value)
            space_before = int(token.get_tag('space_before').value)
            # print((char, pred, space_before))
            segments.append((char, pred, space_before))
        text = merge2(segments, text)
        # for token in text.tokens:
        #     print(token.save())

        # convert to InputExamples
        example = self._convert_to_example(text)
        # examples = read_examples_from_file(args.data_dir, 'test')
        result, predictions, probs = self.tagger2.predict([example])
        # print(result)
        # print('predictions', predictions)
        # print('probs', probs)
        text = self.merge3(predictions[0], text, probs=probs[0])

        for token in text.tokens:
            lemma=self.lemmatizer.lemmatize(token)
            token.predicted_lemma=lemma
            #find such interp and set as disamb
            found_interp=False
            for interp in token.interpretations:
                if interp.tag==token.predicted and interp.lemma==token.predicted_lemma:
                    interp.disamb=True
                    found_interp=True
                    break
            if not found_interp:
                interp=KInterpretation(token.predicted_lemma, token.predicted, disamb=True, manual=True)
                token.add_interpretation(interp)
            
            # print(f'{token.form}\t{token.predicted}\t{lemma}')
        return text

    def tag_texts(self, texts: [KText]):
        sentences = [self.flair_converter.convert_to_sentence(text) for text in texts]
        self.tagger.predict(sentences, mini_batch_size=self.tokenizer_batch_size)
        # TODO

    def _convert_to_example(self, text: KText) -> InputExample:
        words = [token.form for token in text.tokens]
        labels = ['O'] * len(words)
        # print('len(words)', len(words))  # 296; [(0, 0, 206, 283), (15, 206, 296, 296)]

        return InputExample('X', words=words, labels=labels)

    def merge3(self, predictions, text: KText, probs=None) -> KText:
        # print(len(predictions), len(text.tokens))
        assert len(predictions) == len(text.tokens)  # 412 296
        if probs is not None:
            assert len(predictions) == len(probs)
            for pred, prob, token in zip(predictions, probs, text.tokens):
                token.predicted = pred
                token.predicted_score=prob
        else:
            for pred, token in zip(predictions, text.tokens):
                token.predicted = pred

        return text


if __name__ == '__main__':
    parser = ArgumentParser(description='')
    parser.add_argument('path', help='path to DAG')
    parser.add_argument('tokenizer_path', help='path to tokenizer model')
    parser.add_argument('--lemmatizer_path', default=None, help='path to lemmatizer stats')
    parser.add_argument('--tokenizer_batch_size', type=int, default=32, help='tokenizer batch size')
    parser.add_argument('--korba', action='store_true', help='korba format')
    parser.add_argument('--no_cuda', action='store_true', help='dont use CUDA')
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=1, help='dont change')  # TODO
    parser.add_argument('--model_type', default='xlmroberta', help='type of tagger transformer model')
    parser.add_argument('--model_name_or_path', default='', help='path to tagger transformer model')
    parser.add_argument('--cache_dir', default='', help='')

    args = parser.parse_args()

    print(args)

    kftt = KFTT(tokenizer_path=args.tokenizer_path,
                tagger_path=args.model_name_or_path,
                model_type=args.model_type,
                no_cuda=args.no_cuda,
                cache_dir=args.cache_dir,
                per_gpu_eval_batch_size=args.per_gpu_eval_batch_size,
                lemmatizer_path=args.lemmatizer_path,
                tokenizer_batch_size=args.tokenizer_batch_size)

    for ktext in convert_to_ktagger(args.path, corpus='', korba=args.korba, only_disamb=False):
        print(ktext)
        kftt.tag(ktext)

#TODO neural lemmatizer json
#TODO korba format output as separate script?