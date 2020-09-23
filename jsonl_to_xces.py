from argparse import ArgumentParser

import jsonlines

from ktagger import KText
from shortest_path import shortest_path

parser = ArgumentParser(description='Converts disamb JSONL to gold XCES')
parser.add_argument('disamb_path', help='path to disamb JSONL')
parser.add_argument('output_path', help='path to output DAG')
parser.add_argument('--sentences', action='store_true', help='split to sentences')
parser.add_argument('--shortest', action='store_true', help='output shortest path without disamb')

args = parser.parse_args()


def escape_xml(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace('\'',
                                                                                                            '&apos;')


with jsonlines.open(args.disamb_path) as reader, open(args.output_path, 'w') as writer:
    writer.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    writer.write('<!DOCTYPE cesAna SYSTEM "xcesAnaIPI.dtd">\n')
    writer.write('<cesAna xmlns:xlink="http://www.w3.org/1999/xlink" version="1.0" type="lex disamb">\n')
    writer.write('<chunkList>\n')
    

    
    for data in reader:
        ktext = KText.load(data)
        
        writer.write(' <chunk type="p">\n')
        
        tokens = sorted(ktext.tokens, key=lambda t: (t.start_offset, t.end_offset))

        if args.shortest:
            tokens=shortest_path(ktext)

        i = 0
        writer.write('  <chunk type="s">\n')
        for token in tokens:
            # print(token.form)
            if args.shortest:
                if token.manual:
                    continue
            else:
                if not token.has_disamb():
                    continue
            tags = set([interpretation.tag for interpretation in token.interpretations if
                        not interpretation.manual])
            poss = set([tag.split(':', 1)[0] for tag in tags])

            if not token.space_before:
                writer.write('   <ns/>\n')
            writer.write('   <tok>\n')
            
            writer.write('    <orth>%s</orth>\n' % escape_xml(token.form))
            lemma='X'

            for interp in token.interpretations:
                if args.shortest and interp.manual: continue
                if interp.disamb:
                    writer.write('    <lex disamb="1"><base>%s</base><ctag>%s</ctag></lex>\n' % (
                    escape_xml(interp.lemma), interp.tag))
                
                #if disamb then write second time
                if not interp.manual:
                    writer.write('    <lex><base>%s</base><ctag>%s</ctag></lex>\n' % (escape_xml(interp.lemma), interp.tag))
            # else:
            #     writer.write('    <lex disamb="1"><base>%s</base><ctag>%s</ctag></lex>\n' % (escape_xml(lemma), token.disamb_tag()))
            #     if 'ign' in tags:
            #         writer.write('    <lex><base>%s</base><ctag>%s</ctag></lex>\n' % (escape_xml(token.form), 'ign'))
            writer.write('   </tok>\n')
            i += 1

            if args.sentences and token.sentence_end:
                writer.write('  </chunk>\n')
                writer.write('  <chunk type="s">\n')

        writer.write('  </chunk>\n')
        writer.write(' </chunk>\n')
    writer.write('</chunkList>\n')
    writer.write('</cesAna>\n')