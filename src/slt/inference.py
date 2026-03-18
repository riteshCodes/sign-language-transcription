import sys

import torch
import wandb

sys.path.append(".")

from src.slt.configs import BLEU_N_GRAM, DEVICE
from src.slt.metrics import SLTmetricBLEU, SLTmetricROUGE


def infer(model, feature_extractor, dataloader):

    model.to(DEVICE)

    model.eval()
    results = []

    val_bleu_metric = SLTmetricBLEU(n_gram=BLEU_N_GRAM)
    val_rouge_metric = SLTmetricROUGE() # ROUGE-L metrics

    table = wandb.Table(columns=["Ground Truth", "Prediction"])

    with torch.no_grad():
        for (input_frames, target_sentences, valid_frames) in dataloader:
            input_frames = input_frames.to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                # Extract DINOv2 features
                feat_tokens = feature_extractor.extract_features(input_frames)
                predicted_sentences = model(dinov2_features=feat_tokens, valid_frames=valid_frames)  # model returns the decoded text

            references_val = [[ref] for ref in target_sentences]  # Each candidate has one reference here
            bleu_score = val_bleu_metric.compute_score(predicted_sentences, references_val)
            if bleu_score is None:
                bleu_score = 0.0
            rouge_score = val_rouge_metric.compute_score(predicted_sentences, references_val)
            
            print(f"[Inference] - BLEU-{BLEU_N_GRAM} Score = {bleu_score}, ROUGE-L Score: {rouge_score}")
                
            results.append(
                {
                    "Ground-Truth": target_sentences,
                    "Prediction": predicted_sentences
                }
            )
            table.add_data(target_sentences, predicted_sentences)

    print("Prediction Results:")
    for item in results:
        print(f"Ground Truth: {item['Ground-Truth']}")
        print(f"Predicted:    {item['Prediction']}\n")

    wandb.log({"Inference Results": table})