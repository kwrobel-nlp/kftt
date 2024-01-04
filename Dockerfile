FROM huggingface/transformers-pytorch-gpu:4.27.4

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app

COPY . .

CMD python3 korba_predict.py "tests/data4/ann_morphosyntax_ambig.dag" model/tokenizer.pt --korba --model_name_or_path model/tagger/ --lemmatizer_path model/lemmas.json