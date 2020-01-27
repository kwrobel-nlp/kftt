from pathlib import Path
from typing import List

import torch

import flair
from flair.data import Sentence, Token
from flair.embeddings import TokenEmbeddings, replace_with_language_code
from flair.file_utils import cached_path


class FlairEmbeddingsEnd(TokenEmbeddings):
    """Contextual string embeddings of words, as proposed in Akbik et al., 2018.
    Modified to return backward model embeddings at the end of a word (in the same place of sentence as forward)."""

    def __init__(self, model, fine_tune: bool = False, chars_per_chunk: int = 512):
        """
        initializes contextual string embeddings using a character-level language model.
        :param model: model string, one of 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',
                'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward'
                depending on which character language model is desired.
        :param fine_tune: if set to True, the gradient will propagate into the language model. This dramatically slows down
                training and often leads to overfitting, so use with caution.
        :param  chars_per_chunk: max number of chars per rnn pass to control speed/memory tradeoff. Higher means faster but requires
                more memory. Lower means slower but less memory.
        """
        super().__init__()

        cache_dir = Path("embeddings")

        aws_path: str = "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources"

        self.PRETRAINED_MODEL_ARCHIVE_MAP = {
            # multilingual models
            "multi-forward": f"{aws_path}/embeddings-v0.4.3/lm-jw300-forward-v0.1.pt",
            "multi-backward": f"{aws_path}/embeddings-v0.4.3/lm-jw300-backward-v0.1.pt",
            "multi-v0-forward": f"{aws_path}/embeddings-v0.4/lm-multi-forward-v0.1.pt",
            "multi-v0-backward": f"{aws_path}/embeddings-v0.4/lm-multi-backward-v0.1.pt",
            "multi-v0-forward-fast": f"{aws_path}/embeddings-v0.4/lm-multi-forward-fast-v0.1.pt",
            "multi-v0-backward-fast": f"{aws_path}/embeddings-v0.4/lm-multi-backward-fast-v0.1.pt",
            # English models
            "en-forward": f"{aws_path}/embeddings-v0.4.1/big-news-forward--h2048-l1-d0.05-lr30-0.25-20/news-forward-0.4.1.pt",
            "en-backward": f"{aws_path}/embeddings-v0.4.1/big-news-backward--h2048-l1-d0.05-lr30-0.25-20/news-backward-0.4.1.pt",
            "en-forward-fast": f"{aws_path}/embeddings/lm-news-english-forward-1024-v0.2rc.pt",
            "en-backward-fast": f"{aws_path}/embeddings/lm-news-english-backward-1024-v0.2rc.pt",
            "news-forward": f"{aws_path}/embeddings-v0.4.1/big-news-forward--h2048-l1-d0.05-lr30-0.25-20/news-forward-0.4.1.pt",
            "news-backward": f"{aws_path}/embeddings-v0.4.1/big-news-backward--h2048-l1-d0.05-lr30-0.25-20/news-backward-0.4.1.pt",
            "news-forward-fast": f"{aws_path}/embeddings/lm-news-english-forward-1024-v0.2rc.pt",
            "news-backward-fast": f"{aws_path}/embeddings/lm-news-english-backward-1024-v0.2rc.pt",
            "mix-forward": f"{aws_path}/embeddings/lm-mix-english-forward-v0.2rc.pt",
            "mix-backward": f"{aws_path}/embeddings/lm-mix-english-backward-v0.2rc.pt",
            # Arabic
            "ar-forward": f"{aws_path}/embeddings-stefan-it/lm-ar-opus-large-forward-v0.1.pt",
            "ar-backward": f"{aws_path}/embeddings-stefan-it/lm-ar-opus-large-backward-v0.1.pt",
            # Bulgarian
            "bg-forward-fast": f"{aws_path}/embeddings-v0.3/lm-bg-small-forward-v0.1.pt",
            "bg-backward-fast": f"{aws_path}/embeddings-v0.3/lm-bg-small-backward-v0.1.pt",
            "bg-forward": f"{aws_path}/embeddings-stefan-it/lm-bg-opus-large-forward-v0.1.pt",
            "bg-backward": f"{aws_path}/embeddings-stefan-it/lm-bg-opus-large-backward-v0.1.pt",
            # Czech
            "cs-forward": f"{aws_path}/embeddings-stefan-it/lm-cs-opus-large-forward-v0.1.pt",
            "cs-backward": f"{aws_path}/embeddings-stefan-it/lm-cs-opus-large-backward-v0.1.pt",
            "cs-v0-forward": f"{aws_path}/embeddings-v0.4/lm-cs-large-forward-v0.1.pt",
            "cs-v0-backward": f"{aws_path}/embeddings-v0.4/lm-cs-large-backward-v0.1.pt",
            # Danish
            "da-forward": f"{aws_path}/embeddings-stefan-it/lm-da-opus-large-forward-v0.1.pt",
            "da-backward": f"{aws_path}/embeddings-stefan-it/lm-da-opus-large-backward-v0.1.pt",
            # German
            "de-forward": f"{aws_path}/embeddings/lm-mix-german-forward-v0.2rc.pt",
            "de-backward": f"{aws_path}/embeddings/lm-mix-german-backward-v0.2rc.pt",
            "de-historic-ha-forward": f"{aws_path}/embeddings-stefan-it/lm-historic-hamburger-anzeiger-forward-v0.1.pt",
            "de-historic-ha-backward": f"{aws_path}/embeddings-stefan-it/lm-historic-hamburger-anzeiger-backward-v0.1.pt",
            "de-historic-wz-forward": f"{aws_path}/embeddings-stefan-it/lm-historic-wiener-zeitung-forward-v0.1.pt",
            "de-historic-wz-backward": f"{aws_path}/embeddings-stefan-it/lm-historic-wiener-zeitung-backward-v0.1.pt",
            # Spanish
            "es-forward": f"{aws_path}/embeddings-v0.4/language_model_es_forward_long/lm-es-forward.pt",
            "es-backward": f"{aws_path}/embeddings-v0.4/language_model_es_backward_long/lm-es-backward.pt",
            "es-forward-fast": f"{aws_path}/embeddings-v0.4/language_model_es_forward/lm-es-forward-fast.pt",
            "es-backward-fast": f"{aws_path}/embeddings-v0.4/language_model_es_backward/lm-es-backward-fast.pt",
            # Basque
            "eu-forward": f"{aws_path}/embeddings-stefan-it/lm-eu-opus-large-forward-v0.2.pt",
            "eu-backward": f"{aws_path}/embeddings-stefan-it/lm-eu-opus-large-backward-v0.2.pt",
            "eu-v1-forward": f"{aws_path}/embeddings-stefan-it/lm-eu-opus-large-forward-v0.1.pt",
            "eu-v1-backward": f"{aws_path}/embeddings-stefan-it/lm-eu-opus-large-backward-v0.1.pt",
            "eu-v0-forward": f"{aws_path}/embeddings-v0.4/lm-eu-large-forward-v0.1.pt",
            "eu-v0-backward": f"{aws_path}/embeddings-v0.4/lm-eu-large-backward-v0.1.pt",
            # Persian
            "fa-forward": f"{aws_path}/embeddings-stefan-it/lm-fa-opus-large-forward-v0.1.pt",
            "fa-backward": f"{aws_path}/embeddings-stefan-it/lm-fa-opus-large-backward-v0.1.pt",
            # Finnish
            "fi-forward": f"{aws_path}/embeddings-stefan-it/lm-fi-opus-large-forward-v0.1.pt",
            "fi-backward": f"{aws_path}/embeddings-stefan-it/lm-fi-opus-large-backward-v0.1.pt",
            # French
            "fr-forward": f"{aws_path}/embeddings/lm-fr-charlm-forward.pt",
            "fr-backward": f"{aws_path}/embeddings/lm-fr-charlm-backward.pt",
            # Hebrew
            "he-forward": f"{aws_path}/embeddings-stefan-it/lm-he-opus-large-forward-v0.1.pt",
            "he-backward": f"{aws_path}/embeddings-stefan-it/lm-he-opus-large-backward-v0.1.pt",
            # Hindi
            "hi-forward": f"{aws_path}/embeddings-stefan-it/lm-hi-opus-large-forward-v0.1.pt",
            "hi-backward": f"{aws_path}/embeddings-stefan-it/lm-hi-opus-large-backward-v0.1.pt",
            # Croatian
            "hr-forward": f"{aws_path}/embeddings-stefan-it/lm-hr-opus-large-forward-v0.1.pt",
            "hr-backward": f"{aws_path}/embeddings-stefan-it/lm-hr-opus-large-backward-v0.1.pt",
            # Indonesian
            "id-forward": f"{aws_path}/embeddings-stefan-it/lm-id-opus-large-forward-v0.1.pt",
            "id-backward": f"{aws_path}/embeddings-stefan-it/lm-id-opus-large-backward-v0.1.pt",
            # Italian
            "it-forward": f"{aws_path}/embeddings-stefan-it/lm-it-opus-large-forward-v0.1.pt",
            "it-backward": f"{aws_path}/embeddings-stefan-it/lm-it-opus-large-backward-v0.1.pt",
            # Japanese
            "ja-forward": f"{aws_path}/embeddings-v0.4.1/lm__char-forward__ja-wikipedia-3GB/japanese-forward.pt",
            "ja-backward": f"{aws_path}/embeddings-v0.4.1/lm__char-backward__ja-wikipedia-3GB/japanese-backward.pt",
            # Dutch
            "nl-forward": f"{aws_path}/embeddings-stefan-it/lm-nl-opus-large-forward-v0.1.pt",
            "nl-backward": f"{aws_path}/embeddings-stefan-it/lm-nl-opus-large-backward-v0.1.pt",
            "nl-v0-forward": f"{aws_path}/embeddings-v0.4/lm-nl-large-forward-v0.1.pt",
            "nl-v0-backward": f"{aws_path}/embeddings-v0.4/lm-nl-large-backward-v0.1.pt",
            # Norwegian
            "no-forward": f"{aws_path}/embeddings-stefan-it/lm-no-opus-large-forward-v0.1.pt",
            "no-backward": f"{aws_path}/embeddings-stefan-it/lm-no-opus-large-backward-v0.1.pt",
            # Polish
            "pl-forward": f"{aws_path}/embeddings/lm-polish-forward-v0.2.pt",
            "pl-backward": f"{aws_path}/embeddings/lm-polish-backward-v0.2.pt",
            "pl-opus-forward": f"{aws_path}/embeddings-stefan-it/lm-pl-opus-large-forward-v0.1.pt",
            "pl-opus-backward": f"{aws_path}/embeddings-stefan-it/lm-pl-opus-large-backward-v0.1.pt",
            # Portuguese
            "pt-forward": f"{aws_path}/embeddings-v0.4/lm-pt-forward.pt",
            "pt-backward": f"{aws_path}/embeddings-v0.4/lm-pt-backward.pt",
            # Pubmed
            "pubmed-forward": f"{aws_path}/embeddings-v0.4.1/pubmed-2015-fw-lm.pt",
            "pubmed-backward": f"{aws_path}/embeddings-v0.4.1/pubmed-2015-bw-lm.pt",
            # Slovenian
            "sl-forward": f"{aws_path}/embeddings-stefan-it/lm-sl-opus-large-forward-v0.1.pt",
            "sl-backward": f"{aws_path}/embeddings-stefan-it/lm-sl-opus-large-backward-v0.1.pt",
            "sl-v0-forward": f"{aws_path}/embeddings-v0.3/lm-sl-large-forward-v0.1.pt",
            "sl-v0-backward": f"{aws_path}/embeddings-v0.3/lm-sl-large-backward-v0.1.pt",
            # Swedish
            "sv-forward": f"{aws_path}/embeddings-stefan-it/lm-sv-opus-large-forward-v0.1.pt",
            "sv-backward": f"{aws_path}/embeddings-stefan-it/lm-sv-opus-large-backward-v0.1.pt",
            "sv-v0-forward": f"{aws_path}/embeddings-v0.4/lm-sv-large-forward-v0.1.pt",
            "sv-v0-backward": f"{aws_path}/embeddings-v0.4/lm-sv-large-backward-v0.1.pt",
            # Tamil
            "ta-forward": f"{aws_path}/embeddings-stefan-it/lm-ta-opus-large-forward-v0.1.pt",
            "ta-backward": f"{aws_path}/embeddings-stefan-it/lm-ta-opus-large-backward-v0.1.pt",
        }

        if type(model) == str:

            # load model if in pretrained model map
            if model.lower() in self.PRETRAINED_MODEL_ARCHIVE_MAP:
                base_path = self.PRETRAINED_MODEL_ARCHIVE_MAP[model.lower()]
                model = cached_path(base_path, cache_dir=cache_dir)

            elif replace_with_language_code(model) in self.PRETRAINED_MODEL_ARCHIVE_MAP:
                base_path = self.PRETRAINED_MODEL_ARCHIVE_MAP[
                    replace_with_language_code(model)
                ]
                model = cached_path(base_path, cache_dir=cache_dir)

            elif not Path(model).exists():
                raise ValueError(
                    f'The given model "{model}" is not available or is not a valid path.'
                )

        from flair.models import LanguageModel

        if type(model) == LanguageModel:
            self.lm: LanguageModel = model
            self.name = f"Task-LSTM-{self.lm.hidden_size}-{self.lm.nlayers}-{self.lm.is_forward_lm}"
        else:
            self.lm: LanguageModel = LanguageModel.load_language_model(model)
            self.name = str(model)

        # embeddings are static if we don't do finetuning
        self.fine_tune = fine_tune
        self.static_embeddings = not fine_tune

        self.is_forward_lm: bool = self.lm.is_forward_lm
        self.chars_per_chunk: int = chars_per_chunk

        # embed a dummy sentence to determine embedding_length
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token("hello"))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(
            embedded_dummy[0].get_token(1).get_embedding()
        )

        # set to eval mode
        self.eval()

    def train(self, mode=True):

        # make compatible with serialized models (TODO: remove)
        if "fine_tune" not in self.__dict__:
            self.fine_tune = False
        if "chars_per_chunk" not in self.__dict__:
            self.chars_per_chunk = 512

        if not self.fine_tune:
            pass
        else:
            super(FlairEmbeddingsEnd, self).train(mode)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        # gradients are enable if fine-tuning is enabled
        gradient_context = torch.enable_grad() if self.fine_tune else torch.no_grad()

        with gradient_context:

            # if this is not possible, use LM to generate embedding. First, get text sentences
            text_sentences = [sentence.to_plain_string() for sentence in sentences]

            start_marker = "\n"
            end_marker = " "

            # get hidden states from language model
            all_hidden_states_in_lm = self.lm.get_representation(
                text_sentences, start_marker, end_marker, self.chars_per_chunk
            )

            if not self.fine_tune:
                all_hidden_states_in_lm = all_hidden_states_in_lm.detach()

            # take first or last hidden states from language model as word representation
            for i, sentence in enumerate(sentences):
                sentence_text = sentence.to_plain_string()

                offset_forward: int = len(start_marker) - 1
                offset_backward: int = len(sentence_text) + len(start_marker) - 1

                for token in sentence.tokens:

                    offset_forward += len(token.text)
                    offset_backward -= len(token.text)

                    if self.is_forward_lm:
                        offset = offset_forward
                    else:
                        offset = offset_backward

                    embedding = all_hidden_states_in_lm[offset, i, :]

                    if token.whitespace_after:
                        offset_forward += 1
                        offset_backward -= 1



                    # only clone if optimization mode is 'gpu'
                    if flair.embedding_storage_mode == "gpu":
                        embedding = embedding.clone()

                    token.set_embedding(self.name, embedding)

            del all_hidden_states_in_lm

        return sentences

    def __str__(self):
        return self.name