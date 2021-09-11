#! /usr/bin/env python3

import argparse
import sys
from pathlib import Path
from collections import defaultdict

argparser = argparse.ArgumentParser(description="""
Evaluates PolEval 2020 Task 2
""",
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
argparser.add_argument('evaldir', help="directory with tagger results in DAG format")
argparser.add_argument('golddir', help="directory with gold standard DAG files")
args = argparser.parse_args()

class ValidationException(Exception):
    def __init__(self, message):
        self.message = message

def paragraphs(filepath):
    "Generate paraghraphs as lists of lines"
    with filepath.open('r') as infile:
        lines = []
        for line in infile:
            line = line.rstrip('\r\n')
            if line == '':
                if len(lines)>0:
                    yield lines
                lines = []
            elif line[0] == '#':
                pass
            else:
                lines.append(line)
        if len(lines)>0:
            yield lines

class Rating:
    def __init__(self):
        self.num_tokens = 0
        self.ign_tokens = 0
        self.manual_known = 0
        self.manual_ign = 0
        self.correct_tokens = 0
        self.correct_ign = 0
        self.correct_manual_known = 0
        self.correct_manual_ign = 0
        self.correct_segmentation = 0
        self.correct_pos = 0

    def step_tokens(self, ign=False, manual=False):
        self.num_tokens += 1
        if ign:
            self.ign_tokens += 1
        if manual:
            if ign:
                self.manual_ign +=1
            else:
                self.manual_known +=1

    def step_correct(self, ign=False, manual=False):
        self.correct_tokens += 1
        if ign:
            self.correct_ign += 1
        if manual:
            if ign:
                self.correct_manual_ign +=1
            else:
                self.correct_manual_known +=1

    def __str__(self):
        return """\

Accuracy (Your score!): {}

Tokens total:           {}
Correct tokens:         {}
Unknown tokens:         {} ({:.2f}%)
Correct unknown:        {}
Accuracy on unknown:    {}
Known tokens:           {} ({:.2f}%)
Accuracy on known:      {}
Manual tokens:          {} (known {} + ign {})
Correct manual:         {}
Accuracy on manual:     {}
Accuracy manual known:  {}
Accuracy manual ign:    {}
Segmentation:           {}
Correct POS:            {}
""".format(self.correct_tokens/self.num_tokens,
           self.num_tokens,
           self.correct_tokens,
           self.ign_tokens, self.ign_tokens/self.num_tokens*100,
           self.correct_ign,
           self.correct_ign/self.ign_tokens if self.ign_tokens else 0,
           self.num_tokens-self.ign_tokens, (self.num_tokens-self.ign_tokens)/self.num_tokens*100,
           (self.correct_tokens-self.correct_ign)/(self.num_tokens-self.ign_tokens),
           self.manual_known+self.manual_ign, self.manual_known, self.manual_ign,
           self.correct_manual_ign+self.correct_manual_known,
           (self.correct_manual_ign+self.correct_manual_known)/(self.manual_known+self.manual_ign) if self.manual_known+self.manual_ign else 0,
           self.correct_manual_known/self.manual_known if self.manual_known else 0,
           self.correct_manual_ign/self.manual_ign if self.manual_ign else 0,
           self.correct_segmentation/self.num_tokens,
           self.correct_pos/self.num_tokens,
           
           )

class MorphInterp:
    def __init__(self, dagline):
        try:
            row=dagline.split('\t')
            #print(row)
            if len(row)==7:
                (start, stop, self.orth, self.lemma, self.tag, self.nps, disamb) = row
            elif len(row)==12: #korba
                (start, stop, self.orth, self.lemma, self.tag, _,_,_,_,_,self.nps, disamb) = row
            self.orth=self.orth.replace(' ','')
        except ValueError:
            print(dagline, file=sys.stderr)
            sys.exit()
        self.start = int(start)
        self.stop = int(stop)
        self.msd = self.tag.split(':')
        self.pos = self.tag.split(':')[0]
        if not disamb in ['', 'disamb', 'disamb_manual']:
            raise ValidationException('Wrong value in 7th column: ‘{}’'
                                      .format(disamb))
        self.isdisamb = disamb != ''
        self.ismanual = disamb == 'disamb_manual'

    def __repr__(self):
        return '‹{}–{} {} {} {} {}›'.format(self.start, self.stop, self.orth, self.lemma, self.tag, '+' if self.isdisamb else '−')

class MorphDag:
    def __init__(self, par):
        self.dag = defaultdict(list)
        self.chosen = []
        for line in par:
            interp = MorphInterp(line)
            self.dag[interp.start].append(interp)
            if interp.isdisamb:
                self.chosen.append(interp)

    def next_chosen(self, start_pos):
        "Find the chosen interpretation starting at start_pos"
        cand = [c for c in self.dag[start_pos] if c.isdisamb]
        if len(cand) != 1:
            raise ValidationException('No chosen interpretation starting at position {}!'.format(start_pos))
        return cand[0]
        

    def validate_against(self, gold, rating):
        e_current = 0
        e_start = self.chosen[e_current].start
        e_text = ''
        g_text = ''
        aligned = True # are current positions aligned in eval and gold?
        for gi in gold.chosen:
            # is gi unknown to the morphological analyser?:
            is_ign = len([i for i in gold.dag[gi.start]
                          if i.stop == gi.stop and i.tag == 'ign']) > 0
            rating.step_tokens(ign=is_ign, manual=gi.ismanual)
            
            ei = self.next_chosen(e_current)
#DEBUG            print(ei, '⇔', gi, aligned)

            if len(ei.orth) != len(gi.orth):
                aligned = False
            
            if aligned:
                e_current = ei.stop
                if ei.orth == gi.orth: #correct segmentation?
                    rating.correct_segmentation += 1
                if ei.orth == gi.orth and ei.tag.split(':')[0] == gi.tag.split(':')[0]:
                    rating.correct_pos += 1
                    
                if ei.orth == gi.orth and ei.tag == gi.tag:
                    rating.step_correct(ign=is_ign, manual=gi.ismanual)
                else:
                    print(ei.orth , gi.orth , ei.tag , gi.tag, file=sys.stderr)
            else:
                # print('ABC')
                g_text += gi.orth
                while len(e_text)+len(ei.orth) <= len(g_text):
                    e_text += ei.orth
                    print('OMG', e_text, g_text, file=sys.stderr)
                    if g_text[:len(e_text)] != e_text:
                        raise ValidationException('Token {} is inconsistent with underlying text!'.format(ei))
                    e_current = ei.stop
                    if len(e_text) == len(g_text):
                        aligned = True
                        e_text = ''
                        g_text = ''
                        break
                    ei = self.next_chosen(e_current)

goldpaths = list(Path(args.golddir).glob('*.dag'))
if len(goldpaths) == 0:
    raise ValidationException('No .dag files were found under {}'
                              .format(args.golddir))
        
print('''
Poleval 2020 Task 2

Evaluating files in: {}
against {} gold standard files in: {}
'''
      .format(args.evaldir, len(goldpaths), args.golddir))

rating = Rating()

for goldpath in sorted(goldpaths):
    evalpath = Path(args.evaldir).joinpath(goldpath.name)
    if not evalpath.exists():
        raise ValidationException('Missing file in results: {}'.format(goldpath.name))
    print('File: {}'.format(goldpath.name), file=sys.stderr)
    evalpars = list(paragraphs(evalpath))
    goldpars = list(paragraphs(goldpath))
    if not len(evalpars) == len(goldpars):
        raise ValidationException('Misaligned paragraphs in file {}: eval file contains {} paragraph(s) while gold-standard expects {}'.format(goldpath.name, len(evalpars), len(goldpars)))
    for evalpar, goldpar in zip(evalpars, goldpars):
        evaldag = MorphDag(evalpar)
        golddag = MorphDag(goldpar)
        evaldag.validate_against(golddag, rating)

print(rating)
