from typing import List, Tuple
import wandb
import evaluate
import os
from torcheval.metrics import BLEUScore


class SLTmetricBLEU:
    def __init__(self, n_gram=4):
        self.n_gram = n_gram
        self.bleu_metric = BLEUScore(n_gram=self.n_gram)

    def compute_score(self, candidate_sentences, reference_sentences):
        """
        Args:
            predicted_sentences (str)
            target_sentences (str)

        Returns:
            float: BLEU score between 0 and 1
        """

        # Reset the metric value
        self.bleu_metric.reset()

        if len(candidate_sentences) != len(reference_sentences):
            raise ValueError(
                "Number of predicted and target sentences must be equal")

        # Validate all candidates and references
        candidate_sentences, reference_sentences = filter_by_word_count(candidates=candidate_sentences,
                                                      references=reference_sentences,
                                                      word_count=self.n_gram)

        if candidate_sentences and reference_sentences:
            self.bleu_metric.update(candidate_sentences, reference_sentences)

            # Compute and return the BLEU score
            score = self.bleu_metric.compute()
            return score.item()
        else:
            return None


def filter_by_word_count(candidates,
                         references,
                         word_count: int) -> Tuple[List[str], List[List[str]]]:

    filtered_preds = []
    filtered_refs = []

    for candidate, reference in zip(candidates, references):
        candidate_word_list = candidate.strip().split()
        if len(candidate_word_list) >= word_count:
            filtered_preds.append(candidate.strip())
            if isinstance(reference, list):
                filtered_refs.append(reference)
            else:
                filtered_refs.append([reference])
    return filtered_preds, filtered_refs


class SLTmetricROUGE:
    def __init__(self):
        self.rouge = evaluate.load("rouge", experiment_id = f'job_{os.getpid()}')

    def compute_score(self, predicted_sentences, reference_sentences):

        if len(predicted_sentences) != len(reference_sentences):
            raise ValueError(
                "Number of predicted and target sentences must be equal")

        results = self.rouge.compute(predictions=predicted_sentences,
                                     references=reference_sentences,
                                     use_stemmer=True)
        """
        Valid rouge types:
            "rouge1": unigram (1-gram) based scoring
            "rouge2": bigram (2-gram) based scoring
            "rougeL": Longest common subsequence based scoring.
            "rougeLSum": splits text using "\n"
        """
        return results["rougeL"] # return type is numpy.float64

def setup_wandb_metrics(bleu_ngram=4):
    wandb.define_metric("train_step")
    wandb.define_metric("val_step")
    wandb.define_metric("validation_count")

    # Training metrics
    wandb.define_metric("Cross-Entropy Loss", step_metric="train_step")
    wandb.define_metric("Perplexity", step_metric="train_step")
    wandb.define_metric(f"BLEU-{bleu_ngram} Score", step_metric="train_step")
    wandb.define_metric("ROUGE-L Score", step_metric="train_step")
    wandb.define_metric("Training: Average Cross-Entropy Loss", step_metric="train_step")
    wandb.define_metric("Training: Average Perplexity", step_metric="train_step")
    wandb.define_metric("Learning Rate", step_metric="train_step")

    # Validation metrics for within validation steps
    wandb.define_metric("Validation: Cross-Entropy Loss", step_metric="val_step")
    wandb.define_metric("Validation: Perplexity", step_metric="val_step")
    wandb.define_metric(f"Validation: BLEU-{bleu_ngram} Score", step_metric="val_step")
    wandb.define_metric("Validation: ROUGE-L Score", step_metric="val_step")

    # After completion of every validation
    wandb.define_metric("Validation: Average Cross-Entropy Loss", step_metric="validation_count")
    wandb.define_metric("Validation: Average Perplexity", step_metric="validation_count")
