"""Merges output form Morfeusz woth reference data."""
import collections
from argparse import ArgumentParser

import jsonlines
import sys

from ktagger import KText, KToken, KInterpretation
from utils2 import jsonlines_gzip_reader

parser = ArgumentParser(description='Train')
parser.add_argument('jsonl_path', help='path to JSONL for getting text')
parser.add_argument('--century', default=None, help='')

args = parser.parse_args()

igns=[]
texts=0
masks=0
number_of_tokens=[]
no_interps=0
more_than_one_disamb=0
no_disamb=0
disamb=0
more_than_one_disamb_but_all_same_tags=0
unique_tags=set()
more_than_one_disamb_but_all_same_tags_list=collections.defaultdict(int)
intepretations = 0
manual =0 
manual_interps =0 
intepretations_of_disamb_token=0

with jsonlines_gzip_reader(args.jsonl_path) as reader:
    for data in reader:
        ktext = KText.load(data)
        corpus = ktext.id.split('▁')[1].split('_')[0]
        if args.century is not None and corpus!=args.century:
            continue
        texts+=1
        number_of_tokens.append(len(ktext.tokens))
        for token in ktext.tokens:
            if len(token.interpretations)==0:
                no_interps+=1
                print(token.save())
            if not token.has_disamb():
                no_disamb+=1
                #print(token.form)
                # print('X', ktext.id, token.form, [i.save() for i in token.interpretations])
            else:
                disamb+=1
                intepretations_of_disamb_token += len(token.interpretations)
            if token.manual:
                manual+=1
            if len([interpretation for interpretation in token.interpretations if interpretation.disamb])>1:
                more_than_one_disamb+=1
                
                if len(set([interpretation.tag for interpretation in token.interpretations if interpretation.disamb]))==1:
                    more_than_one_disamb_but_all_same_tags+=1
                    print('MORE THAN 1 DISAMB WITH SAME TAGS', token.save(), file=sys.stderr)
                else:
                    pass
                    print('MORE THAN 1 DISAMB WITH DIFFERENT TAGS', token.save(), file=sys.stderr)
                    more_than_one_disamb_but_all_same_tags_list['+'.join(sorted([i.tag for i in token.interpretations]))]+=1
                    # print(token.save())

            intepretations+=len(token.interpretations)
            for interp in token.interpretations:
                if interp.tag=='ign':
                    igns+=(token.form,)
                elif interp.tag=='MASK':
                    masks+=1

                if interp.disamb:
                    unique_tags.add(interp.tag)
                    
                if interp.manual:
                    manual_interps+=1

stats={}
stats['name']=args.jsonl_path
stats['ign']=len(igns)
stats['texts']=texts
stats['number of tokens']=sum(number_of_tokens)
stats['avg number of tokens']=sum(number_of_tokens)/len(number_of_tokens)
stats['no_interps']=no_interps
stats['more_than_one_disamb']=more_than_one_disamb
stats['more_than_one_disamb_but_all_same_tags']=more_than_one_disamb_but_all_same_tags
stats['no_disamb']=no_disamb
stats['disamb']=disamb
stats['masks']=masks
stats['unique tags']=len(unique_tags)
stats['interpretations']=intepretations
stats['interpretations_per_token']=intepretations/sum(number_of_tokens)
stats['manual']=manual
stats['manual_interps']=manual_interps
stats['interps_of_disamb_token']=intepretations_of_disamb_token
stats['interps_per_disamb_token']=intepretations_of_disamb_token/disamb if disamb else 0

for k,v in stats.items():
    print(k,'\t', v)

# print('Name','\t', args.jsonl_path)
# print('ign:', len(igns))
# # print(igns)
# print('Texts:', texts)
# print('number of tokens', sum(number_of_tokens))
# print('avg number of tokens', sum(number_of_tokens)/len(number_of_tokens))
# print('no_interps:', no_interps)
# print('more_than_one_disamb:', more_than_one_disamb)
# print('more_than_one_disamb_but_all_same_tags:', more_than_one_disamb_but_all_same_tags)
# print('no_disamb:', no_disamb)
# print('disamb:', disamb)
# print('masks:', masks)
# print('unique tags:', len(unique_tags))
# print('interpretations:', intepretations, intepretations/sum(number_of_tokens))
# print('manual:', manual)
# print('manual_interps:', manual_interps)
# print('interps_per_token')
# print('interps_per_disamb_token')
# print(more_than_one_disamb_but_all_same_tags_list)

for k,v in sorted(more_than_one_disamb_but_all_same_tags_list.items(), key=lambda x: x[1]):
    print(v, k, file=sys.stderr)