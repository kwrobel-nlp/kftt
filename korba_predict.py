import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='seqeval')

import glob
from argparse import ArgumentParser

import tqdm

from dag_to_jsonl import convert_to_ktagger
from kftt_predict import KFTT

if __name__ == '__main__':
    parser = ArgumentParser(description='')
    parser.add_argument('path', help='glob pattern to DAGs')
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

    for path in tqdm.tqdm(glob.glob(args.path, recursive=True)):
        writer=open(f'{path}.predicted','w')
        for ktext in convert_to_ktagger(path, corpus='', korba=args.korba, only_disamb=False):
            # print(ktext)
            kftt.tag(ktext)
            
            for i, token in enumerate(ktext.tokens):
                
                col9='manual' if not token.interpretations else '' #manual
                col10='eos' if token.sentence_end else '' #eos
                col11='' if token.space_before else 'nps' #nps
                writer.write('\t'.join([str(i), str(i + 1), token.form, token.predicted_lemma, token.predicted, '','','1.000',col9,col10, col11, 'disamb']))
                writer.write('\n')
            writer.write('\n')
        writer.close()
        
#TODO neural lemmatizer json
