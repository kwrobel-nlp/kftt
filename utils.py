from pathlib import Path
from typing import List, Dict

import torch
import torch.nn
from flair.data import Dictionary, Token
from flair.embeddings import TokenEmbeddings
from flair.models import SequenceTagger
from flair.training_utils import Metric, Result, store_embeddings
from torch.utils.data import DataLoader


class BinarySequenceTagger(SequenceTagger):
    def __init__(
            self,
            hidden_size: int,
            embeddings: TokenEmbeddings,
            tag_dictionary: Dictionary,
            tag_type: str,
            use_crf: bool = True,
            use_rnn: bool = True,
            rnn_layers: int = 1,
            dropout: float = 0.0,
            word_dropout: float = 0.05,
            locked_dropout: float = 0.5,
            train_initial_hidden_state: bool = False,
            rnn_type: str = "LSTM",
            pickle_module: str = "pickle",
            beta: float = 1.0,
            loss_weights: Dict[str, float] = None,
            class_name: str = '1'
    ):
        """
        Initializes a SequenceTagger
        :param hidden_size: number of hidden states in RNN
        :param embeddings: word embeddings used in tagger
        :param tag_dictionary: dictionary of tags you want to predict
        :param tag_type: string identifier for tag type
        :param use_crf: if True use CRF decoder, else project directly to tag space
        :param use_rnn: if True use RNN layer, otherwise use word embeddings directly
        :param rnn_layers: number of RNN layers
        :param dropout: dropout probability
        :param word_dropout: word dropout probability
        :param locked_dropout: locked dropout probability
        :param train_initial_hidden_state: if True, trains initial hidden state of RNN
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for classes (tags) for the loss function
        (if any tag's weight is unspecified it will default to 1.0)

        """

        super().__init__(hidden_size, embeddings, tag_dictionary, tag_type, use_crf, use_rnn, rnn_layers, dropout,
                         word_dropout, locked_dropout, train_initial_hidden_state, rnn_type, pickle_module, beta,
                         loss_weights)
        self.class_name = class_name

    def evaluate(
            self,
            data_loader: DataLoader,
            out_path: Path = None,
            embedding_storage_mode: str = "none",
    ) -> (Result, float):

        if type(out_path) == str:
            out_path = Path(out_path)

        with torch.no_grad():
            eval_loss = 0

            batch_no: int = 0

            metric = Metric("Evaluation", beta=self.beta)

            lines: List[str] = []

            if self.use_crf:
                transitions = self.transitions.detach().cpu().numpy()
            else:
                transitions = None

            for batch in data_loader:
                batch_no += 1

                with torch.no_grad():
                    features = self.forward(batch)
                    loss = self._calculate_loss(features, batch)
                    tags, _ = self._obtain_labels(
                        feature=features,
                        batch_sentences=batch,
                        transitions=transitions,
                        get_all_tags=False,
                    )

                eval_loss += loss

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token: Token = token
                        token.add_tag_label("predicted", tag)

                        # append both to file for evaluation
                        eval_line = "{} {} {} {}\n".format(
                            token.text,
                            token.get_tag(self.tag_type).value,
                            tag.value,
                            tag.score,
                        )
                        lines.append(eval_line)
                    lines.append("\n")
                for sentence in batch:
                    # make list of gold tags
                    gold_tags = [
                        (tag.tag, str(tag)) for tag in sentence.get_spans(self.tag_type)
                    ]
                    # make list of predicted tags
                    predicted_tags = [
                        (tag.tag, str(tag)) for tag in sentence.get_spans("predicted")
                    ]

                    # check for true positives, false positives and false negatives
                    for tag, prediction in predicted_tags:
                        if (tag, prediction) in gold_tags:
                            metric.add_tp(tag)
                        else:
                            metric.add_fp(tag)

                    for tag, gold in gold_tags:
                        if (tag, gold) not in predicted_tags:
                            metric.add_fn(tag)
                        else:
                            metric.add_tn(tag)

                store_embeddings(batch, embedding_storage_mode)

            eval_loss /= batch_no

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            detailed_result = (
                f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
                f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
                f"\nBINARY: f1-score {metric.f_score(self.class_name)}"
            )
            for class_name in metric.get_classes():
                detailed_result += (
                    f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                    f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
                    f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                    f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
                    f"{metric.f_score(class_name):.4f}"
                )

            result = Result(
                main_score=metric.f_score(self.class_name),
                log_line=f"{metric.precision(self.class_name)}\t{metric.recall(self.class_name)}\t{metric.f_score(self.class_name)}",
                log_header="PRECISION\tRECALL\tF1",
                detailed_results=detailed_result,
            )

            return result, eval_loss
