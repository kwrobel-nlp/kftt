
import collections
import sys
from argparse import ArgumentParser

import jsonlines
import os 

from ktagger import KText, KToken, KInterpretation
from utils2 import jsonlines_gzip_reader

from dag_to_jsonl import read_dag

parser = ArgumentParser(description='Adds lemma')
parser.add_argument('plain_path', help='path to plain JSONL')
parser.add_argument('pred_path', help='path to tagger predictions DAG')
parser.add_argument('output_path', help='path to output TSV')
args = parser.parse_args()

def lemmatize(token:KToken, plain_token:KToken=None):
    form = token.form
    predicted_tag = token.interpretations[0].tag
    
    if plain_token is not None:
        #1. sprawdz czy sa interpretacje z takim samym tagiem
        # print(token.form, token.interpretations[0].tag)
        
        matched_lemmas=[]
        for interp in plain_token.interpretations:
            # print(interp.lemma, interp.tag)
            if predicted_tag==interp.tag:
                matched_lemmas.append(interp.lemma)
                # print('-', interp.lemma)
                
        matched_lemmas=list(set(matched_lemmas))
        if len(matched_lemmas)==1:
            return matched_lemmas[0]
        elif len(matched_lemmas)>1:
            print('More than 1 matched lemma:', form, predicted_tag, matched_lemmas)
            return matched_lemmas[0] #TODO może zwracać obie?
        
        #2. lemat dla tagu o największej części wspólnej od początku, przy czym cz. mowy musi być taka sama?
        predicted_pos=predicted_tag.split(':')[0]
        prefixes=collections.defaultdict(list)
        with_other_pos=[]
        for interp in plain_token.interpretations:
            if interp.tag.startswith(predicted_pos):
                # print(form, predicted_tag, interp.lemma, interp.tag)
                prefix=os.path.commonprefix([predicted_tag, interp.tag])
                prefixes[len(prefix)].append(interp.lemma)
            else:
                # print('Different POS:', form, predicted_tag, interp.lemma, interp.tag)
                with_other_pos.append(interp)
        # print(prefixes)
        if prefixes:
            most_common=list(set(max(prefixes.items())[1]))
            if len(most_common)==1:
                return most_common[0]
            else:
                print('More than 1 lemma with most common tag:', form, predicted_tag, most_common)
                print([(interp.tag, interp.lemma) for interp in plain_token.interpretations])
                return most_common[0] #TODO
            
        if not prefixes and with_other_pos: #brak interpretacjiz taką samą cz. mowy ale istnieją z inną
            non_ign_lemmas=[]
            for interp in with_other_pos:
                print('Only with different POS:', form, predicted_tag, interp.lemma, interp.tag) #TODO
                if interp.tag!='ign':
                    non_ign_lemmas.append(interp.lemma)
            if non_ign_lemmas:
                return non_ign_lemmas[0] #TODO

    else:
        print('No plain_token:', form, predicted_tag)
        
    return form
    
    #TODO wielkośc liter?
    #TODO: inna segmentacja powoduje brak interpretacji? jeśli tokeny są złączone

with jsonlines_gzip_reader(args.plain_path) as reader, open(args.output_path, 'w') as writer:
    # for p in read_dag(args.pred_path):
    #     print(p)
    # odtworzyć offsety w dagu i na ich podstawie matchować interpretacje
    for data, data_pred in zip(reader, read_dag(args.pred_path)):
        ktext = KText.load(data)
        ktext.tokens = sorted(ktext.tokens, key=lambda t: (t.start_position, t.end_position))

        # ktext2 = KText.load(data_pred)
        # print(data)
        # print(data_pred['tokens'])
        tokens=[]
        for token in data_pred['tokens']:
            ktoken = KToken(token['segment'], token['space_before'], None,
                            None, None, token['start_position'], token['end_position'],
                            None)
            #TODO sentence end, space_before brakuje
            
            ktoken.interpretations = [KInterpretation(interpretation['lemma'], interpretation['tag'], interpretation['disamb'], None) for interpretation in token['interpretations']]
            tokens.append(ktoken)

        ktext_pred = KText(None)
        ktext_pred.tokens=tokens
        
        ktext_pred.text=ktext_pred.infer_original_text()
        ktext_pred.fix_offsets3()
        
        # for token in ktext_pred.tokens:
        #     print(token.form, token.start_offset2, token.end_offset2)

        plain={}
        ktext.fix_offsets3()
        for token in ktext.tokens:
            # print(token.form, token.start_offset2, token.end_offset2)
            assert (token.start_offset2, token.end_offset2) not in plain
            plain[(token.start_offset2, token.end_offset2)]=token
            
        for token in ktext_pred.tokens:
            print(token.form, token.start_offset2, token.end_offset2)
            if (token.start_offset2, token.end_offset2) in plain:
                plain_token=plain[(token.start_offset2, token.end_offset2)]
                lemma=lemmatize(token, plain_token)
            else:
                lemma=lemmatize(token)
            print(lemma)
        # break
        
#TODO policzyć w korpusie (form, tag, lemma)
# pokazuje 513 521 More than 1 matched lemma: pokazuje fin:sg:ter:imperf ['pokazować', 'pokazywać']
# mają 898 902 More than 1 matched lemma: mają fin:pl:ter:imperf ['mieć', 'maić']
# że 1193 1195 More than 1 matched lemma: że part ['że', 'ż']
#TODO policzyć w korpusie (form, lemma)
# niż 866 869
# Only with different POS: niż comp nizać impt:sg:sec:imperf
# Only with different POS: niż comp niż conj
# Only with different POS: niż comp niż prep:nom
# Only with different POS: niż comp niż subst:sg:acc:m
# Only with different POS: niż comp niż subst:sg:nom:m
# Only with different POS: niż comp niża subst:pl:gen:f
# Only with different POS: niż comp niżyć impt:sg:sec:imperf
#TODO policzyć w korpusie (lemma)
#TODO policzyć w korpusie (form) - z nieoznaczonego

# drugich 852 859
# More than 1 lemma with most common tag: drugich subst:pl:acc:manim1 ['drugie', 'druga', 'drugi']
# [('subst:pl:gen:f', 'druga'), ('subst:pl:loc:f', 'druga'), ('adjnum:pl:acc:manim1:pos', 'drugi'), ('adjnum:pl:gen:m:pos', 'drugi'), ('adjnum:pl:gen:f:pos', 'drugi'), ('adjnum:pl:gen:n:pos', 'drugi'), ('adjnum:pl:loc:m:pos', 'drugi'), ('adjnum:pl:loc:f:pos', 'drugi'), ('adjnum:pl:loc:n:pos', 'drugi'), ('subst:pl:gen:m', 'drugi'), ('subst:pl:loc:m', 'drugi'), ('subst:pl:gen:n', 'drugie'), ('subst:pl:loc:n', 'drugie')]

# niemający 637 646
# More than 1 lemma with most common tag: niemający pact:sg:nom:m:imperf:aff:pos ['mieć', 'maić']

#python3 lemmatize.py test_lemma/korba3-korba_reczna-plain.jsonl_0_test test_lemma/test_predictions_long.txt.dag temp | less