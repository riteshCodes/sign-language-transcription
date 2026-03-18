import os
import sys
from datetime import datetime

import torch

sys.path.append(".")

from src.slt.configs import CHECKPOINTS_PATH, DEVICE


def save_training_checkpoint(model, optimizer, epoch, loss):
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{model.__class__.__name__}_checkpoint_{timestamp}_epoch_{epoch}_loss_{loss}.pth"
    filepath = os.path.join(CHECKPOINTS_PATH, filename)

    print(f"Saving model checkpoint: {filename}")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to '{filepath}'")
    return filename


def load_checkpoint(model, checkpoint_model=None, optimizer=None):
    if checkpoint_model is None:
        raise Exception("Checkpoint model is not specified")
    
    filepath = os.path.join(CHECKPOINTS_PATH, checkpoint_model)

    checkpoint = torch.load(filepath, map_location=DEVICE)

    model.to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Checkpoint loaded from '{checkpoint_model}'")
    print(f"Average Loss of checkpoint {checkpoint_model}: {loss}")

    return model, optimizer, loss


def load_validated_checkpoint(model, validated_checkpoint_model=None):
    if validated_checkpoint_model is None:
        raise Exception("Checkpoint model is not specified")
    
    filepath = os.path.join(CHECKPOINTS_PATH, validated_checkpoint_model)

    checkpoint = torch.load(filepath, map_location=DEVICE)
    print(f"Checkpoint loaded from '{validated_checkpoint_model}'")

    model.to(DEVICE)
    model.load_state_dict(checkpoint)
    
    return model


class EarlyStoppingBLEU:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_bleu_score = float('-inf') # since higher score is better for BLEU
        self.early_stop = False

    def __call__(self, bleu_score, model):
        if bleu_score > self.best_bleu_score - self.delta:
            self.best_bleu_score = bleu_score
            self.counter = 0
            os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"{model.__class__.__name__}_validated_checkpoint_{timestamp}_bleu_{bleu_score}.pth"
            filepath = os.path.join(CHECKPOINTS_PATH, filename)
            torch.save(model.state_dict(), filepath)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class EarlyStoppingCE:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_ce_loss = float('inf')
        self.early_stop = False

    def __call__(self, ce_loss, model):
        if ce_loss < self.best_ce_loss - self.delta:
            self.best_ce_loss = ce_loss
            self.counter = 0
            os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"{model.__class__.__name__}_validated_checkpoint_{timestamp}_ce_loss_{ce_loss}.pth"
            filepath = os.path.join(CHECKPOINTS_PATH, filename)
            torch.save(model.state_dict(), filepath)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True