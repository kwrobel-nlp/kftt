import json
import sys
from typing import List

import regex


class KInterpretation:
    def __init__(self, lemma: str, tag: str, disamb: bool, manual: bool = None):
        self.lemma = lemma
        self.tag = tag
        self.disamb = disamb
        self.manual: bool = manual

    @staticmethod
    def load(data):
        kinterpretation = KInterpretation(data['lemma'], data['tag'], data['disamb'], data['manual'])
        return kinterpretation

    def save(self):
        d = {'lemma': self.lemma, 'tag': self.tag, 'disamb': self.disamb,
             'manual': self.manual, }
        return d

    @staticmethod
    def load_compact(data):
        return KInterpretation(*data)

    def save_compact(self):
        return [self.lemma, self.tag, self.disamb, self.manual]


class KToken:
    def __init__(self,
                 form: str,
                 space_before: bool,
                 start_offset: int,
                 end_offset: int,
                 sentence_end: bool = None,
                 start_position: int = None,
                 end_position: int = None,
                 manual: bool = False
                 ):
        self.form: str = form
        self.interpretations: List[KInterpretation] = []
        self.space_before = space_before
        self.sentence_end: int = sentence_end
        self.start_offset: int = start_offset
        self.end_offset: int = end_offset
        self.start_position: int = start_position
        self.end_position: int = end_position
        self.manual: bool = manual  # manual segmentation

    def add_interpretation(self, interpretation: KInterpretation):
        self.interpretations.append(interpretation)

    @staticmethod
    def load(data):
        ktoken = KToken(data['form'], data['space_before'], data['start_offset'],
                        data['end_offset'], data['sentence_end'], data['start_position'], data['end_position'],
                        data['manual'])
        ktoken.interpretations = [KInterpretation.load(interpretation) for interpretation in data['interpretations']]
        return ktoken

    def save(self):
        d = {'form': self.form, 'space_before': self.space_before, 'sentence_end': self.sentence_end,
             'start_offset': self.start_offset, 'end_offset': self.end_offset, 'start_position': self.start_position,
             'end_position': self.end_position, 'manual': self.manual}
        d['interpretations'] = [token.save() for token in self.interpretations]
        return d

    @staticmethod
    def load_compact(data):
        ktoken = KToken(*data[:-1])
        ktoken.interpretations = [KInterpretation.load_compact(d) for d in data[-1]]
        return ktoken

    def save_compact(self):
        return [self.form, self.space_before, self.start_offset, self.end_offset,
                self.sentence_end, self.start_position, self.end_position,
                [interpretation.save_compact() for interpretation in self.interpretations]]

    def has_disamb(self):
        return any([interpretation.disamb for interpretation in self.interpretations])

    def disamb_tag(self):
        disamb_tags=[interpretation.tag for interpretation in self.interpretations if interpretation.disamb]
        if len(disamb_tags)!=1:
            print('WARNING Disamb tags', len(disamb_tags), self.has_disamb(), self.save(), file=sys.stderr)
        return disamb_tags[0]

class KText:
    """Represents paragraph."""

    def __init__(self, paragraph_id):
        self.id: str = paragraph_id
        self.text: str = None
        self.tokens: List[KToken] = []
        self.year: int = None

    def __repr__(self):
        return f"{self.text} {len(self.tokens)}"

    def add_token(self, token: KToken):
        self.tokens.append(token)

    def add_reference_token(self, reference_token: KToken):
        # assert len(reference_token.interpretations) == 1
        if len(reference_token.interpretations) != 1:
            if len(reference_token.interpretations) == 0:
                ki = KInterpretation(reference_token.form, 'MASK', disamb=True, manual=True)
                reference_token.add_interpretation(ki)
            else:
                print('More than one disamb', len(reference_token.interpretations), file=sys.stderr)
            # return

        reference_interpretation = reference_token.interpretations[0]
        assert reference_interpretation.disamb == True
        # 1. find token with the same lemma and positions
        found_token = False
        for token in self.tokens:
            # print(token.save())
            if token.form == reference_token.form \
                    and token.start_offset == reference_token.start_offset \
                    and token.end_offset == reference_token.end_offset:
                # print('Found token', file=sys.stderr)
                # 2. find interpretation
                token.sentence_end = reference_token.sentence_end
                found_interpretation = False
                for interpretation in token.interpretations:
                    if interpretation.lemma == reference_interpretation.lemma \
                            and interpretation.tag == reference_interpretation.tag:
                        # print('Found interp', file=sys.stderr)
                        interpretation.disamb = True
                        if reference_interpretation.manual != False:  # or reference_interpretation.tag == 'MASK'
                            pass
                            # print('WARNING interpretation not marked as manual', file=sys.stderr)
                        interpretation.manual = False
                        found_interpretation = True
                        break
                if not found_interpretation:
                    # print('NOT Found interp', file=sys.stderr)
                    assert reference_interpretation.manual == True
                    token.interpretations.append(reference_interpretation)
                found_token = True
                break
        if not found_token:
            # print('NOT Found token', reference_token.save(), file=sys.stderr)
            if reference_interpretation.manual != True:
                print('NOT Found token without manual marking', reference_token.save(), file=sys.stderr)
                reference_interpretation.manual = True

            # choose one disamb interpretation if multiple disambs
            reference_token.interpretations = reference_token.interpretations[0:1]

            self.tokens.append(reference_token)
            reference_token.manual = True

    def save2(self):
        return json.dumps(self, ensure_ascii=False, indent=1, default=lambda x: x.__dict__, sort_keys=True)

    def save(self):
        d = {'id': self.id, 'text': self.text, 'year': self.year}
        d['tokens'] = [token.save() for token in self.tokens]
        return d

    @staticmethod
    def load(data):
        # print(data)
        ktext = KText(data['id'])
        ktext.text = data['text']
        ktext.year = data['year']
        ktext.tokens = [KToken.load(token) for token in data['tokens']]
        return ktext

    @staticmethod
    def load_compact(data):
        ktext = KText(*data[:-1])
        ktext.tokens = [KToken.load_compact(d) for d in data[-1]]
        return ktext

    def save_compact(self):  # two times smaller
        return [self.id, self.text, self.year, [token.save_compact() for token in self.tokens]]

    def infer_original_text(self):
        """ Bierze pod uwagę pierwszą interpretację z danego węzła. Problem gdy analizator dodaje spacje, któ©ych nie ma."""
        strings = []
        last_position = 0
        for token in self.tokens:
            if token.start_position == last_position:
                if token.space_before:
                    strings.append(' ')
                strings.append(token.form)
                last_position = token.end_position

        return ''.join(strings)

    def check_offsets(self):
        text = self.infer_original_text()
        for token in self.tokens:
            if token.start_offset is None or token.end_offset is None: continue
            # assert token.form == text[token.start_offset:token.end_offset]
            if token.form != text[token.start_offset:token.end_offset]:
                print('ERROR TOKEN OFFSET', f"{token.form} {text[token.start_offset:token.end_offset]}_",
                      file=sys.stderr)

    def fix_offsets(self, text: str):
        """Only for disambiguated text (with unambiguous segmentation)."""
        last_offset = 0
        for token in self.tokens:
            if token.end_offset is not None:
                assert last_offset < token.end_offset
            form = token.form
                    
            try:
                start_offset = text.index(form, last_offset)
            except:
                print('Form not found in text:', form, text, last_offset, file=sys.stderr)
            end_offset = start_offset + len(form)

            token.start_offset = start_offset
            token.end_offset = end_offset

            last_offset = end_offset



    def find_ambiguous_end_offsets(self):
        ambiguous_end_offsets = set()
        for token in self.tokens:
            if token.end_offset is None:
                continue
            for token2 in self.tokens:
                if token2.start_offset is None or token2.end_offset is None:
                    continue
                if token2.start_offset < token.end_offset < token2.end_offset:
                    ambiguous_end_offsets.add(token.end_offset)
                    token.ambiguous=True
                    break
        return ambiguous_end_offsets

    def fix_offsets2(self):
        text = self.text
        start_offsets = {}
        end_offsets = {0: 0}
        for token in self.tokens:
            start_position = token.start_position
            end_position = token.end_position
            # print(start_position, end_position)
            try:
                previous_end_offset = end_offsets[start_position]
                if text[previous_end_offset:previous_end_offset + len(token.form)] == token.form:
                    token.space_before = False
                elif text[previous_end_offset + 1:previous_end_offset + 1 + len(token.form)] == token.form:
                    token.space_before = True
                    previous_end_offset += 1

                start_offsets[start_position] = previous_end_offset
                token.start_offset = previous_end_offset

                if text[previous_end_offset:previous_end_offset + len(token.form)] == token.form:
                    end_offsets[end_position] = previous_end_offset + len(token.form)
                    token.end_offset = previous_end_offset + len(token.form)
                else:  # manually corrected tokenization introducing space before
                    print('OMITTING token with different space before',
                          text[previous_end_offset:previous_end_offset + len(token.form)], token.form,
                          file=sys.stderr)
                    token.start_offset = None
                    token.end_offset = None
            except KeyError:
                print('OMITTING node without incoming edges', token.form, file=sys.stderr)
                token.start_offset = None
                token.end_offset = None

        # print(offsets.values())
        del end_offsets[0]
        
        
    def fix_offsets3(self):
        """Against text without spaces"""
        text = regex.sub('\s+', '', self.text)
        start_offsets = {}
        end_offsets = {0: 0}
        for token in self.tokens:
            start_position = token.start_position
            end_position = token.end_position
            # print(start_position, end_position)
            try:
                previous_end_offset = end_offsets[start_position]
                if text[previous_end_offset:previous_end_offset + len(token.form)] == token.form:
                    # token.space_before = False
                    pass
                elif text[previous_end_offset + 1:previous_end_offset + 1 + len(token.form)] == token.form:
                    # token.space_before = True
                    previous_end_offset += 1

                start_offsets[start_position] = previous_end_offset
                token.start_offset2 = previous_end_offset

                if text[previous_end_offset:previous_end_offset + len(token.form)] == token.form:
                    end_offsets[end_position] = previous_end_offset + len(token.form)
                    token.end_offset2 = previous_end_offset + len(token.form)
                else:  # manually corrected tokenization introducing space before
                    print('OMITTING token with different space before',
                          text[previous_end_offset:previous_end_offset + len(token.form)], token.form,
                          file=sys.stderr)
                    token.start_offset2 = None
                    token.end_offset2 = None
            except KeyError:
                print('OMITTING node without incoming edges', token.form, file=sys.stderr)
                token.start_offset2 = None
                token.end_offset2 = None

        # print(offsets.values())
        del end_offsets[0]
