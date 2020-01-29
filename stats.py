"""Merges output form Morfeusz woth reference data."""
from argparse import ArgumentParser

import jsonlines


from ktagger import KText, KToken, KInterpretation


parser = ArgumentParser(description='Train')
parser.add_argument('jsonl_path', help='path to JSONL for getting text')

args = parser.parse_args()

igns=[]
masks=0
number_of_tokens=[]
no_interps=0
more_than_one_disamb=0
no_disamb=0
more_than_one_disamb_but_all_same_tags=0
with jsonlines.open(args.jsonl_path) as reader:
    for data in reader:
        ktext = KText.load(data)
        number_of_tokens.append(len(ktext.tokens))
        for token in ktext.tokens:
            if len(token.interpretations)==0:
                no_interps+=1
                print(token.save())
            if not token.has_disamb():
                no_disamb+=1
            if len([interpretation for interpretation in token.interpretations if interpretation.disamb])>1:
                more_than_one_disamb+=1
                
                if len(set([interpretation.tag for interpretation in token.interpretations if interpretation.disamb]))==1:
                    more_than_one_disamb_but_all_same_tags+=1
                else:
                    pass
                    print(token.save())
                    # print(token.save())
                    
                
            for interp in token.interpretations:
                if interp.tag=='ign':
                    igns+=(token.form,)
                elif interp.tag=='MASK':
                    masks+=1

print('ign:', len(igns))
# print(igns)
print('avg number of tokens', sum(number_of_tokens)/len(number_of_tokens))
print('no_interps:', no_interps)
print('more_than_one_dismab:', more_than_one_disamb)
print('more_than_one_disamb_but_all_same_tags:', more_than_one_disamb_but_all_same_tags)
print('no_disamb:', no_disamb)
print('masks:', masks)
