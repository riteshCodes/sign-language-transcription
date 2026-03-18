import os
import sys

from torch.utils.data import DataLoader, Subset, random_split

import wandb

sys.path.append(".")

from dotenv import load_dotenv

from src.slt.bart import SLTmodelBART
from src.slt.configs import (DEVICE, FRAMES_PER_SECOND, TEST_CSV_PATH,
                                  VIDEO_DIR, TRAIN_CSV_PATH,
                                  VIDEO_DIR, VALIDATION_CSV_PATH,
                                  VIDEO_DIR)
from src.slt.custom_dataset import FramesAndTextDataset, collate_fn
from src.slt.dinov2_feature_extractor import DinoV2FeatureExtractor
from src.slt.inference import infer
from src.slt.smollm import SLTmodelSmolLM
from src.slt.train import evaluate_model_inf, training_loop
from src.slt.transforms import test_transform, train_transform
from src.slt.utils import load_checkpoint, load_validated_checkpoint
from src.slt.metrics import setup_wandb_metrics
load_dotenv()

def train_test_SLT_model(slt_model,
                         feature_extractor_model,
                         checkpoint_filename=None):
    
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT")
    ):
        setup_wandb_metrics()
        print("Training Dataset: ")
        training_dataset = FramesAndTextDataset(VIDEO_DIR,
                                                TRAIN_CSV_PATH,
                                                transform=train_transform,
                                                frames_per_second=FRAMES_PER_SECOND)
        # Total training dataset size = 31047 videos

        print("Validation Dataset: ")
        validation_dataset = FramesAndTextDataset(VIDEO_DIR,
                                                  VALIDATION_CSV_PATH,
                                                  transform=test_transform,
                                                  frames_per_second=FRAMES_PER_SECOND)

        print("Test Dataset: ")
        test_dataset = FramesAndTextDataset(VIDEO_DIR,
                                            TEST_CSV_PATH,
                                            transform=test_transform,
                                            frames_per_second=FRAMES_PER_SECOND)

        train_loader = DataLoader(training_dataset, 
                                         batch_size=32,
                                         collate_fn=collate_fn, 
                                         shuffle=True,
                                         drop_last=True,
                                         num_workers=4)
        val_loader = DataLoader(validation_dataset,
                                       batch_size=16,
                                       collate_fn=collate_fn,
                                       drop_last=True,
                                       num_workers=2)
        subset_test_dataset = Subset(test_dataset, 
                                     indices=range(64))
        subset_test_loader = DataLoader(subset_test_dataset,
                                        batch_size=1,
                                        collate_fn=collate_fn,
                                        num_workers=1)

        # Training:
        if checkpoint_filename is not None:
            print(f"[INFO] - Resuming training from Checkpoint: {checkpoint_filename}")
            slt_model, _, _ = load_checkpoint(model=slt_model, checkpoint_model=checkpoint_filename)

        trained_slt_model = training_loop(model=slt_model,
                                          feature_extractor=feature_extractor_model,
                                          train_dataloader=train_loader,
                                          val_dataloader=val_loader)

        # Inference:
        checkpoint_slt_model, _, _ = load_checkpoint(model=slt_model,
                                                     checkpoint_model=trained_slt_model)
        infer(model=checkpoint_slt_model,
              feature_extractor=feature_extractor_model,
              dataloader=subset_test_loader)


def train_test_subset_SLT_model(slt_model,
                         feature_extractor_model):
    
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT")
    ):
        print("Training Dataset: ")
        training_dataset = FramesAndTextDataset(VIDEO_DIR,
                                                TRAIN_CSV_PATH,
                                                transform=train_transform,
                                                frames_per_second=FRAMES_PER_SECOND)
        # Total training dataset size = 31048 videos

        print("Validation Dataset: ")
        validation_dataset = FramesAndTextDataset(VIDEO_DIR,
                                                  VALIDATION_CSV_PATH,
                                                  transform=test_transform,
                                                  frames_per_second=FRAMES_PER_SECOND)

        print("Test Dataset: ")
        test_dataset = FramesAndTextDataset(VIDEO_DIR,
                                            TEST_CSV_PATH,
                                            transform=test_transform,
                                            frames_per_second=FRAMES_PER_SECOND)

        subset_training_dataset = Subset(training_dataset, 
                                         indices=range(64))
        subset_validation_dataset = Subset(validation_dataset,
                                           indices=range(32))
        subset_test_dataset = Subset(test_dataset,
                                     indices=range(8))
        subset_train_loader = DataLoader(subset_training_dataset, 
                                         batch_size=8,
                                         collate_fn=collate_fn,
                                         num_workers=2)
        subset_val_loader = DataLoader(subset_validation_dataset,
                                       batch_size=4,
                                       collate_fn=collate_fn,
                                       num_workers=1)
        subset_test_loader = DataLoader(subset_test_dataset,
                                        batch_size=2,
                                        collate_fn=collate_fn,
                                        num_workers=1)

        # Training:
        trained_slt_model = training_loop(model=slt_model,
                                 feature_extractor=feature_extractor_model,
                                 train_dataloader=subset_train_loader,
                                 val_dataloader=subset_val_loader)

        # Inference:
        checkpoint_slt_model, _, _ = load_checkpoint(model=slt_model,
                                                      checkpoint_model=trained_slt_model)
        infer(model=checkpoint_slt_model,
              feature_extractor=feature_extractor_model,
              dataloader=subset_test_loader)


def inference_with_checkpoint(slt_model, feature_extractor_model, checkpoint_filename):
    
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT")
    ):
        print("Test Dataset: ")
        test_dataset = FramesAndTextDataset(VIDEO_DIR,
                                            TEST_CSV_PATH,
                                            transform=test_transform,
                                            frames_per_second=FRAMES_PER_SECOND)

        subset_test_dataset = Subset(test_dataset, 
                                         indices=range(5))

        subset_test_loader = DataLoader(subset_test_dataset,
                                        batch_size=1,
                                        collate_fn=collate_fn,
                                        num_workers=1)


        print(f"Loading model from checkpoint: {checkpoint_filename}")
        slt_model, _, _ = load_checkpoint(model=slt_model, checkpoint_model=checkpoint_filename)
    
        infer(model=slt_model,
              feature_extractor=feature_extractor_model,
              dataloader=subset_test_loader)
    
def run_validation_with_checkpoint(slt_model, feature_extractor_model, checkpoint_filename):
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    with wandb.init(entity=os.getenv("WANDB_ENTITY"), project=os.getenv("WANDB_PROJECT")):
        print("Validation Dataset:")
        validation_dataset = FramesAndTextDataset(VIDEO_DIR,
                                                  VALIDATION_CSV_PATH,
                                                  transform=test_transform,
                                                  frames_per_second=FRAMES_PER_SECOND)

        val_loader = DataLoader(validation_dataset, batch_size=32, collate_fn=collate_fn, num_workers=4)

        print(f"Loading model from checkpoint: {checkpoint_filename}")
        slt_model, _, _ = load_checkpoint(model=slt_model, checkpoint_model=checkpoint_filename)

        contributing = evaluate_model_inf(
            model=slt_model,
            feature_extractor=feature_extractor_model,
            dataloader=val_loader,
            validation_step=0,
            validation_count=0
        )

if __name__ == "__main__":
    
    dinov2_feature_extractor = DinoV2FeatureExtractor(with_cls_token=True,device=DEVICE)
    smollm_model = SLTmodelSmolLM().to(DEVICE)

    checkpoint_filename = "SLTmodelSmolLM_checkpoint_20250719_0625_epoch_10_loss_0.38342207876337364.pth"
    run_validation_with_checkpoint(smollm_model, dinov2_feature_extractor, checkpoint_filename)
    # inference_with_checkpoint(slt_model=smollm_model,
    #                           feature_extractor_model=dinov2_feature_extractor,
    #                           checkpoint_filename=checkpoint_filename)

    # train_test_subset_SLT_model(slt_model=smollm_model,
    #                      feature_extractor_model=dinov2_feature_extractor)
    
    # train_test_SLT_model(slt_model=smollm_model,
    #                      feature_extractor_model=dinov2_feature_extractor,
    #                      checkpoint_filename=None)
    
    # bart_model = SLTmodelBART().to(DEVICE)
    # train_test_SLT_model(slt_model=bart_model,
    #                      feature_extractor_model=dinov2_feature_extractor)
