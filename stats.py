"""Merges output form Morfeusz woth reference data."""
import collections
from argparse import ArgumentParser

import jsonlines


from ktagger import KText, KToken, KInterpretation


parser = ArgumentParser(description='Train')
parser.add_argument('jsonl_path', help='path to JSONL for getting text')

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

with jsonlines.open(args.jsonl_path) as reader:
    for data in reader:
        ktext = KText.load(data)
        corpus = ktext.id.split('▁')[1].split('_')[0]
        # if corpus!='20':continue
        texts+=1
        number_of_tokens.append(len(ktext.tokens))
        for token in ktext.tokens:
            if len(token.interpretations)==0:
                no_interps+=1
                print(token.save())
            if not token.has_disamb():
                no_disamb+=1
            else:
                disamb+=1
            if len([interpretation for interpretation in token.interpretations if interpretation.disamb])>1:
                more_than_one_disamb+=1
                
                if len(set([interpretation.tag for interpretation in token.interpretations if interpretation.disamb]))==1:
                    more_than_one_disamb_but_all_same_tags+=1
                    print('MORE THAN 1 DISAMB WITH SAME TAGS', token.save())
                else:
                    pass
                    print('MORE THAN 1 DISAMB WITH DIFFERENT TAGS', token.save())
                    more_than_one_disamb_but_all_same_tags_list['+'.join(sorted([i.tag for i in token.interpretations]))]+=1
                    # print(token.save())
                    
                
            for interp in token.interpretations:
                if interp.tag=='ign':
                    igns+=(token.form,)
                elif interp.tag=='MASK':
                    masks+=1

                if interp.disamb:
                    unique_tags.add(interp.tag)

print('ign:', len(igns))
# print(igns)
print('Texts:', texts)
print('number of tokens', sum(number_of_tokens))
print('avg number of tokens', sum(number_of_tokens)/len(number_of_tokens))
print('no_interps:', no_interps)
print('more_than_one_disamb:', more_than_one_disamb)
print('more_than_one_disamb_but_all_same_tags:', more_than_one_disamb_but_all_same_tags)
print('no_disamb:', no_disamb)
print('disamb:', disamb)
print('masks:', masks)
print('unique tags:', len(unique_tags))


print(more_than_one_disamb_but_all_same_tags_list)

for k,v in sorted(more_than_one_disamb_but_all_same_tags_list.items(), key=lambda x: x[1]):
    print(v, k)