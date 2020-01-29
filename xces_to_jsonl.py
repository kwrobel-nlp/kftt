import glob
import logging
from argparse import ArgumentParser
from xml.etree import ElementTree as ET

import jsonlines

from ktagger import KText, KToken, KInterpretation


def read_xces(file_path: str, corpus_name: str='', only_disamb: bool=False):
    paragraphs_defined = True
    ns = False  # no separator
    first_chunk = True

    paragraph_index=0
    for event, elem in ET.iterparse(file_path, events=("start", "end",)):
        if first_chunk and event == "start" and elem.tag in ('chunk', 'sentence'):
            if elem.get('type') == 's' or elem.tag == 'sentence':
                paragraphs_defined = False
            first_chunk = False
        elif event == "end" and elem.tag in ('chunk', 'sentence'):
            xml_sentences = []
            paragraph = KText(f"{corpus_name}▁{file_path}▁{paragraph_index}")
            paragraph_index+=1
            start_position=0
            if paragraphs_defined and elem.tag == 'chunk' and elem.get('type') != 's':
                xml_sentences = elem.getchildren()
            elif (not paragraphs_defined) and (
                    (elem.tag == 'chunk' and elem.get('type') == 's') or elem.tag == 'sentence'):
                xml_sentences = [elem]
            else:
                continue

            for sentence_index, xml_sentence in enumerate(xml_sentences):
                # sentence=Sentence()
                # paragraph.add_sentence(sentence)
                for token_index, xml_token in enumerate(xml_sentence.getchildren()):
                    if xml_token.tag == 'ns':
                        if token_index > 0 or sentence_index > 0:  # omit first ns in paragraph
                            ns = True
                    elif xml_token.tag == 'tok':
                        token = KToken(None, None, None, None, start_position=start_position, end_position=start_position+1)
                        start_position+=1
                        token.space_before = not ns

                        for xml_node in xml_token.getchildren():
                            if xml_node.tag == 'orth':
                                orth = xml_node.text
                                token.form = orth
                            elif xml_node.tag == 'lex':
                                if xml_node.get('disamb') == '1':
                                    disamb = True
                                else:
                                    disamb = False

                                base = xml_node.find('base').text
                                ctag = xml_node.find('ctag').text

                                form = KInterpretation(base, ctag, disamb=False)
                                if disamb:
                                    form.disamb = True
                                    # if token.gold_form is not None:
                                    #     logging.warning(f'More than 1 disamb {file_path} {orth}')
                                    # token.gold_form=form
                                if disamb or not only_disamb:
                                    token.add_interpretation(form)
                            elif xml_node.tag == 'ann':
                                continue
                            else:
                                logging.error('Error 1 {xml_token}')
                        if token.form:
                            paragraph.add_token(token)
                        ns = False
                    else:
                        logging.error(f'Error 2 {xml_token}')
                token.sentence_end = True
            
            paragraph.text = paragraph.infer_original_text()
            paragraph.fix_offsets(paragraph.text)
            yield paragraph
            elem.clear()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file_path', type=str, help='path pattern to XCES corpus')
    parser.add_argument('corpus_name', help='corpus name')
    parser.add_argument('output_path', type=str, help='save path')
    parser.add_argument('--only_disamb', action='store_true',
                        help='save only disamb versions of tokens and interpretations')
    args = parser.parse_args()

    with jsonlines.open(args.output_path, mode='w') as writer:
        for path in sorted(glob.glob(args.file_path)):
            for ktext in read_xces(path, corpus_name=args.corpus_name, only_disamb=args.only_disamb):
                writer.write(ktext.save())
