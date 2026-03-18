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
from src.slt.train import training_loop
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
        
        # subset_validation_dataset = Subset(validation_dataset, indices=range(512))

        subset_test_dataset = Subset(validation_dataset, indices=range(64))

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
        
        test_loader = DataLoader(subset_test_dataset,
                                       batch_size=1,
                                       collate_fn=collate_fn,
                                       drop_last=True,
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
              dataloader=test_loader)


def train_test_subset_SLT_model(slt_model,
                         feature_extractor_model):
    
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
        subset_training_dataset = Subset(training_dataset, indices=range(128))

        print("Validation Dataset: ")
        validation_dataset = FramesAndTextDataset(VIDEO_DIR,
                                                  VALIDATION_CSV_PATH,
                                                  transform=test_transform,
                                                  frames_per_second=FRAMES_PER_SECOND)
        
        subset_validation_dataset = Subset(validation_dataset, indices=range(64))

        subset_test_dataset = Subset(subset_validation_dataset, indices=range(16))

        subset_train_loader = DataLoader(subset_training_dataset, 
                                         batch_size=32,
                                         collate_fn=collate_fn, 
                                         shuffle=True,
                                         drop_last=True,
                                         num_workers=4)
        

        subset_val_loader = DataLoader(subset_validation_dataset,
                                       batch_size=16,
                                       collate_fn=collate_fn,
                                       drop_last=True,
                                       num_workers=2)
        
        subset_test_loader = DataLoader(subset_test_dataset,
                                       batch_size=1,
                                       collate_fn=collate_fn,
                                       drop_last=True,
                                       num_workers=2)

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
    

if __name__ == "__main__":
    
    dinov2_feature_extractor = DinoV2FeatureExtractor(with_cls_token=False, device=DEVICE)
    smollm_model = SLTmodelSmolLM().to(DEVICE)

    # checkpoint_filename = "Basemodel_SmolLM2_360M_2D_Attention.pth"
    
    # inference_with_checkpoint(slt_model=smollm_model,
    #                           feature_extractor_model=dinov2_feature_extractor,
    #                           checkpoint_filename=checkpoint_filename)

    # train_test_subset_SLT_model(slt_model=smollm_model,
    #                      feature_extractor_model=dinov2_feature_extractor)

    train_test_SLT_model(slt_model=smollm_model,
                         feature_extractor_model=dinov2_feature_extractor,
                         checkpoint_filename=None)
    
    # train_test_SLT_model(slt_model=smollm_model,
    #                      feature_extractor_model=dinov2_feature_extractor,
    #                      checkpoint_filename=None)
    
    # bart_model = SLTmodelBART().to(DEVICE)
    # train_test_SLT_model(slt_model=bart_model,
    #                      feature_extractor_model=dinov2_feature_extractor)
