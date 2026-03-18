import sys

import torch
import torch.backends.cudnn as cudnn
# from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, LinearLR,
                                      SequentialLR)
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

import wandb

sys.path.append(".")

from src.slt.configs import (BLEU_N_GRAM, DEVICE, EPOCHS, ETA_MIN,
                                  LOGGING_STEP, LR, PATIENCE, WARM_UP_STEPS)
from src.slt.metrics import SLTmetricBLEU, SLTmetricROUGE
from src.slt.utils import EarlyStoppingCE, save_training_checkpoint

if DEVICE == "cuda":
    cudnn.benchmark = True


def training_loop(model, feature_extractor, train_dataloader, val_dataloader):

    steps_per_epoch = len(train_dataloader)
    # len(train_dataloader): number of steps (batches) per epoch / number of batches produced in one complete iteration over the dataset
    num_training_steps = EPOCHS * steps_per_epoch
    warmup_steps = WARM_UP_STEPS
    
    # steps_per_epoch = math.ceil(num_train_samples / batch_size)

    # validation_logging_step = steps_per_epoch / 2 # validation_logging_step helps to call the validation of the model

    early_stopping_criteria = EarlyStoppingCE(patience=PATIENCE)

    optimizer = AdamW(model.parameters(), lr=LR)  # weight_decay = 0.01 default

    linear_scheduler = LinearLR(optimizer, 
                                start_factor=0.1, 
                                end_factor=1.0, 
                                total_iters=warmup_steps) # Linear warmup
    cosine_scheduler = CosineAnnealingLR(optimizer,
                                         T_max=(num_training_steps - warmup_steps),
                                         eta_min=ETA_MIN) # Cosine annealing
    # Cosine annealing with Linear warmup
    scheduler = SequentialLR(optimizer,
                             schedulers=[linear_scheduler, cosine_scheduler],
                             milestones=[warmup_steps])
    
    # scaler = torch.GradScaler(device=DEVICE)

    global_training_step = 0

    validation_step = 0

    validation_count = 0

    for epoch in range(EPOCHS):
        
        print(f"\n[Training] - Epoch {epoch+1}/{EPOCHS} : ")
        
        if model.training:
            print("[INFO] - Model is in training mode")
        else:
            print("[INFO] - Model switched to training mode")
            model.train()  # Ensure training mode for the model

        # SLT Metrics
        train_bleu_metric = SLTmetricBLEU(BLEU_N_GRAM)  # BLEU-4 metrics
        train_rouge_metric = SLTmetricROUGE() # ROUGE-L metrics

        train_ce_loss = 0
        train_perplexity = 0

        avg_train_ce_loss = 0
        avg_train_perplexity = 0

        # val_ce_loss_list = []

        for (input_frames, target_sentences, valid_frames_counts) in tqdm(train_dataloader, desc="Training"):
            
            input_frames = input_frames.to(DEVICE)
            print(f"[Training] Shape of Input Frames from Dataloader: {input_frames.shape}")
            print(f"[Training] Length of Target Sentences from Dataloader: {len(target_sentences)}")
            # print(f"[Training] Target Sentences from Dataloader: {target_sentences}")
            print(f"[Training] Extracted Valid Frames from Dataloader: {valid_frames_counts}")

            optimizer.zero_grad()

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                # Visual feature extraction
                visual_features = feature_extractor.extract_features(input_frames) # Extract DINOv2 features

                # Model forward pass
                ce_loss, _ , generated_sentences = model(visual_features, target_sentences, valid_frames_counts)

                # BLEU and ROUGE score calculation
            references_train = [[ref] for ref in target_sentences]  # Each candidate has one reference here
            bleu_score = train_bleu_metric.compute_score(generated_sentences, references_train)
            if bleu_score is None: # Cope with higher ngrams if predicted sentence's ngrams count is higher
                bleu_score = 0.0
            rouge_score = train_rouge_metric.compute_score(generated_sentences, references_train)

            train_ce_loss += ce_loss.item()

            perplexity = torch.exp(ce_loss).item() # Perplexity of the model
            train_perplexity += perplexity

            # Scales the loss, and calls backward() to create scaled gradients
            # scaler.scale(ce_loss).backward()
            ce_loss.backward()

            # Gradient clipping to prevent exploding gradients
            # gradient clipping should be applied after scaler.unscale_() and before scaler.step()
            # Unscale the gradients of optimizer's assigned params
            # scaler.unscale_(optimizer)
            # Unscales gradients and calls optimizer.step()
            # scaler.step(optimizer)

            optimizer.step()

            # Updates the scale for next iteration
            # scaler.update()

            if global_training_step % LOGGING_STEP == 0:
                # Get the current learning rate from the optimizer
                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log({
                    "train_step": global_training_step,
                    "Epoch": epoch + 1,
                    "Learning Rate": current_lr,
                    "Cross-Entropy Loss": ce_loss.item(),
                    "Perplexity": perplexity,
                    f"BLEU-{BLEU_N_GRAM} Score": bleu_score,
                    "ROUGE-L Score": rouge_score # Longest common subsequence based scoring
                    },
                )
            
                print(
                    f"[Training] - Step {global_training_step}: LR = {current_lr}, CE-Loss = {ce_loss.item()}, Perplexitiy: {perplexity}")
                print(
                    f"[Training] - Step {global_training_step}: BLEU-{BLEU_N_GRAM} Score = {bleu_score}, ROUGE-L Score: {rouge_score}")

            

            # Scheduler after each batch
            scheduler.step()

            global_training_step += 1

        # val_ce_loss_list.append(val_ce_loss)
            

        # End of Training Dataloader Loop

        # Average losses over all the batches from each epoch
        avg_train_ce_loss = train_ce_loss / len(train_dataloader)
        avg_train_perplexity = train_perplexity / len(train_dataloader)

        # Log all per-epoch metrics
        wandb.log({
            "train_step": global_training_step,
            "Training: Average Cross-Entropy Loss": avg_train_ce_loss,
            "Training: Average Perplexity": avg_train_perplexity
        })

        # Evaluate/Validate the model after every epoch
        avg_val_ce_loss, avg_val_perplexity, validation_step, validation_count = evaluate_model(model=model,
                                                                                                feature_extractor=feature_extractor,
                                                                                                dataloader=val_dataloader, 
                                                                                                validation_step=validation_step,
                                                                                                validation_count=validation_count)
    
        # Early Stopping Check
        early_stopping_criteria(avg_val_ce_loss, model)
        print(f"[Validation] - Current best CE loss of the model : {early_stopping_criteria.best_ce_loss}")
        
        if early_stopping_criteria.early_stop:
            print("[EARLY STOPPING] - Early stopping triggered. Stopping training...")
            break

    # Training ended, Saving model checkpoint
    checkpoint_model = save_training_checkpoint(model=model,
                                                optimizer=optimizer,
                                                loss=avg_train_ce_loss,
                                                epoch=EPOCHS)

    return checkpoint_model


def evaluate_model(model, feature_extractor, dataloader, validation_step, validation_count):

    # Clear GPU to free memory for validation
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    # Validation Phase Metrics
    val_bleu_metric = SLTmetricBLEU(n_gram=BLEU_N_GRAM)
    val_rouge_metric = SLTmetricROUGE() # ROUGE-L metrics

    val_ce_loss = 0
    val_perplexity = 0

    avg_val_ce_loss = 0
    avg_val_perplexity = 0
    
    # Model's training mode
    train_mode = model.training
    if train_mode:
        print("[INFO] - Switching model to evaluation mode")
        model.eval()

    local_step = 0 # local validation steps for logging 
    
    with torch.no_grad():
        for (input_frames, target_sentences, valid_frames) in tqdm(dataloader, desc="Validation"):
            
            input_frames = input_frames.to(DEVICE)
                
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                feat_tokens = feature_extractor.extract_features(input_frames)

                ce_loss, _ , predicted_sentences = model(dinov2_features=feat_tokens,
                                                         valid_frames=valid_frames, 
                                                         target_sentences=target_sentences)
                # print(f"[Validation]: Predicted Sentences: {predicted_sentences}")

            references_val = [[ref] for ref in target_sentences]  # Each candidate has one reference here
            bleu_score = val_bleu_metric.compute_score(predicted_sentences, references_val)
            if bleu_score is None:
                bleu_score = 0.0
            rouge_score = val_rouge_metric.compute_score(predicted_sentences, references_val)
                
            val_ce_loss += ce_loss.item()
                
            perplexity = torch.exp(ce_loss).item() # Perplexity of the model
            val_perplexity += perplexity
            
            if local_step % LOGGING_STEP == 0:
                print(
                    f"[Validation] - Step {local_step}: CE-Loss = {ce_loss.item()}, Perplexitiy: {perplexity}")
                print(
                    f"[Validation] - Step {local_step}: BLEU-{BLEU_N_GRAM} Score = {bleu_score}, ROUGE-L Score: {rouge_score}")
                
                wandb.log({
                    "val_step": validation_step,
                    "Validation: Cross-Entropy Loss": ce_loss.item(),
                    "Validation: Perplexity": perplexity,
                    f"Validation: BLEU-{BLEU_N_GRAM} Score": bleu_score,
                    "Validation: ROUGE-L Score": rouge_score # Longest common subsequence based scoring
                    }
                )
    
            local_step +=1
            validation_step += 1

        # End of Validation Dataloader Loop

        avg_val_ce_loss = val_ce_loss / len(dataloader)
        avg_val_perplexity = val_perplexity / len(dataloader)
        
        print(f"[Validation] - Average Cross-Entropy Loss: {avg_val_ce_loss}")
        print(f"[Validation] - Average Perplexity: {avg_val_perplexity}")
        
        wandb.log({
            "validation_count": validation_count,
            "Validation: Average Cross-Entropy Loss": avg_val_ce_loss,
            "Validation: Average Perplexity": avg_val_perplexity
        })

        validation_count += 1

    # Clear GPU after validation to free memory for next training batch
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Ensure that the model is set to it's original state
    if train_mode:
        model.train()

    return avg_val_ce_loss, avg_val_perplexity, validation_step, validation_count

def evaluate_model_inf(model, feature_extractor, dataloader, validation_step, validation_count):

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    val_bleu_metric = SLTmetricBLEU(n_gram=BLEU_N_GRAM)
    val_rouge_metric = SLTmetricROUGE()

    val_ce_loss = 0
    val_perplexity = 0

    contributing_sentences = []

    train_mode = model.training
    if train_mode:
        print("[INFO] - Switching model to evaluation mode")
        model.eval()

    local_step = 0

    with torch.no_grad():
        for (input_frames, target_sentences, valid_frames) in tqdm(dataloader, desc="Validation"):
            input_frames = input_frames.to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                feat_tokens = feature_extractor.extract_features(input_frames)

                ce_loss, _, predicted_sentences = model(dinov2_features=feat_tokens,
                                                        valid_frames=valid_frames,
                                                        target_sentences=target_sentences)

            references_val = [[ref] for ref in target_sentences]

            # Store BLEU for each pair
            for pred, ref in zip(predicted_sentences, references_val):
                bleu_metric = SLTmetricBLEU(n_gram=BLEU_N_GRAM)
                bleu_score = bleu_metric.compute_score([pred], [ref])
                if bleu_score and bleu_score > 0:
                    contributing_sentences.append({
                        "prediction": pred,
                        "reference": ref[0],
                        "bleu": bleu_score
                    })

            bleu_score = val_bleu_metric.compute_score(predicted_sentences, references_val) or 0.0
            rouge_score = val_rouge_metric.compute_score(predicted_sentences, references_val)
            val_ce_loss += ce_loss.item()
            perplexity = torch.exp(ce_loss).item()
            val_perplexity += perplexity

            if local_step % LOGGING_STEP == 0:
                print(f"[Validation] - Step {local_step}: BLEU-{BLEU_N_GRAM} Score = {bleu_score}")
                wandb.log({
                    "val_step": validation_step,
                    "Validation: BLEU-4 Score": bleu_score,
                    "Validation: Cross-Entropy Loss": ce_loss.item(),
                    "Validation: Perplexity": perplexity,
                    "Validation: ROUGE-L Score": rouge_score
                })

            local_step += 1
            validation_step += 1

    avg_val_ce_loss = val_ce_loss / len(dataloader)
    avg_val_perplexity = val_perplexity / len(dataloader)

    print(f"[Validation] - Average CE Loss: {avg_val_ce_loss:.4f}, Perplexity: {avg_val_perplexity:.4f}")
    print(f"[Validation] - Total contributing sentences with BLEU > 0: {len(contributing_sentences)}\n")

    for idx, item in enumerate(contributing_sentences):
        print(f"BLEU: {item['bleu']:.4f}")
        print(f"Reference : {item['reference']}\n")
        print(f"Prediction: {item['prediction']}")


    wandb.log({
        "validation_count": validation_count,
        "Validation: Average Cross-Entropy Loss": avg_val_ce_loss,
        "Validation: Average Perplexity": avg_val_perplexity
    })

    return contributing_sentences

def train_using_training_arguments(model, train_dataset, val_dataset):
    # Define training arguments for the model
    training_args = TrainingArguments(
        output_dir='./results',          # Directory to save model output and checkpoints
        num_train_epochs=2,              # Number of epochs to train the model
        per_device_train_batch_size=8,   # Batch size per device during training
        per_device_eval_batch_size=8,    # Batch size for evaluation
        warmup_steps=500,                # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # Weight decay for regularization
        logging_dir='./logs',            # Directory to save logs
        logging_steps=10,                # Log metrics every specified number of steps
        evaluation_strategy="epoch",     # Evaluation is done at the end of each epoch
        # Disables reporting to any online services (e.g., TensorBoard, WandB)
        report_to='none'
    )
    # Initializing the Trainer object
    trainer = Trainer(
        # The model to be trained (e.g., our BART model)
        model=model,
        # Training arguments specifying training parameters like learning rate, batch size, etc.
        args=training_args,
        train_dataset=train_dataset,  # The dataset to be used for training the model
        # The dataset to be used for evaluating the model during training
        eval_dataset=val_dataset
    )
    # Starting the training process
    trainer.train()
