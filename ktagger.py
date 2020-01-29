import json
import sys
from typing import List


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


class KToken:
    def __init__(self,
                 form: str,
                 space_before: bool,
                 start_offset: int,
                 end_offset: int,
                 sentence_end: bool = None,
                 start_position: int = None,
                 end_position: int = None
                 ):
        self.form: str = form
        self.interpretations: List[KInterpretation] = []
        self.space_before = space_before
        self.sentence_end: int = sentence_end
        self.start_offset: int = start_offset
        self.end_offset: int = end_offset
        self.start_position: int = start_position
        self.end_position: int = end_position

    def add_interpretation(self, interpretation: KInterpretation):
        self.interpretations.append(interpretation)

    @staticmethod
    def load(data):
        ktoken = KToken(data['form'], data['space_before'], data['start_offset'],
                        data['end_offset'], data['sentence_end'], data['start_position'], data['end_position'], )
        ktoken.interpretations = [KInterpretation.load(interpretation) for interpretation in data['interpretations']]
        return ktoken


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

    def save(self):
        return json.dumps(self, ensure_ascii=False, indent=1, default=lambda x: x.__dict__, sort_keys=True)

    @staticmethod
    def load(data):
        # print(data)
        ktext = KText(data['id'])
        ktext.text = data['text']
        ktext.year = data['year']
        ktext.tokens = [KToken.load(token) for token in data['tokens']]
        return ktext

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