import sys
from typing import Tuple

import torch
import torch.nn as nn
from transformers import (BartForConditionalGeneration, BartTokenizer)
from transformers.modeling_outputs import BaseModelOutput

sys.path.append(".")

from src.slt.configs import (DEVICE, MAX_FRAME_SIZE, MAX_TOKEN_LENGTH, DINOv2_FEATURE_DIMS)


class SLTmodelBART(nn.Module):
    def __init__(self, dinov2_model_name="dinov2_vitl14", bart_model_name="facebook/bart-base"):
        super().__init__()

        """_summary_
        - encoder_hidden_states of BART will be a tuple of tensors, each of shape (batch_size, sequence_length, hidden_size), corresponding to the output of the embedding layer and each encoder layer
        - batch_size: Number of input sequences in the batch and sequence_length: Number of tokens in each input sequence
        - hidden_size: The size of the encoder's hidden representation; for BART-base and BART-large, this is typically 768 or 1024 (config.d_model)
        """
        
        self.device = DEVICE

        # Pretrained BartForConditionalGeneration model
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.bart.to(self.device)

        # Freeze BART Encoder parameters
        for param in self.bart.model.encoder.parameters():
            param.requires_grad = True

        # Ensure BART Decoder parameters are trainable
        for param in self.bart.model.decoder.parameters():
            param.requires_grad = True

        # BART Tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(bart_model_name)
        self._add_special_tokens() # Ensure special tokens are added to the model

        # Adaptive Pooling layer
        # self.pooling_layer = nn.AdaptiveAvgPool1d(output_size=1)

        # Learnable linear layer to project features from DINOv2 to BART (decoder model)
        self.linear_adapter = nn.Linear(DINOv2_FEATURE_DIMS[dinov2_model_name], 
                                        self.bart.config.d_model)

    def _add_special_tokens(self):
        special_tokens_dict = {}
                
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = "<EOS>"
        if not self.tokenizer.bos_token or self.tokenizer.bos_token_id == self.tokenizer.eos_token_id:
            special_tokens_dict["bos_token"] = "<BOS>"
        if not self.tokenizer.pad_token or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            special_tokens_dict["pad_token"] = "[PAD]"

        if special_tokens_dict:
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.smollm.resize_token_embeddings(len(self.tokenizer))

        # self.tokenizer.padding_side = "right"

        print("BOS token:", self.tokenizer.bos_token, "ID:", self.tokenizer.bos_token_id)
        print("EOS token:", self.tokenizer.eos_token, "ID:", self.tokenizer.eos_token_id)
        print("PAD token:", self.tokenizer.pad_token, "ID:", self.tokenizer.pad_token_id)
        print("PADDING side:", self.tokenizer.padding_side)


    def _process_dinov2_features(self, dinov2_features: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Process DINOv2 features to standardize their shape and dimensions.
        Args:
            dinov2_features: Input DINOv2 features
        Returns:
            Tuple of (processed_features, batch_size)
        """

        # print(f"[BART]: Shape of input dinov2_features: {dinov2_features.shape}")
        # Convert to bfloat16 to match BART model dtype
        dinov2_features = dinov2_features.to(dtype=torch.bfloat16, device=self.device)
        
        # Initialization
        batch_and_frame = patch_dim = feat_dim = None
        batch_size = 0

        if dinov2_features.dim() == 2:  # CLS token features
            batch_and_frame, feat_dim = dinov2_features.shape
            batch_size = batch_and_frame // MAX_FRAME_SIZE
            processed_features = dinov2_features.view(batch_size, MAX_FRAME_SIZE, -1)
            # Returns shape of (seq_length, MAX_FRAME_SIZE, feat_dim)
            # print(
            #     f"[BART]: Shape of input dinov2_features [CLS] after reshaping: {processed_features.shape}")

        else:  # Patch token features
            batch_and_frame, patch_dim, feat_dim = dinov2_features.shape  # [(batches*frames), patches, feature_dimension]
            batch_size = batch_and_frame // MAX_FRAME_SIZE
            processed_features = dinov2_features.view(
                batch_size, MAX_FRAME_SIZE, patch_dim, feat_dim
            )
            # Mean Pool over patches and collapse dimension -> Returns shape of (seq_length, MAX_FRAME_SIZE, feat_dim)
            processed_features = processed_features.mean(dim=2)
            # print(
            #     f"[BART]: Shape of input dinov2_features [Patch] after pooling across frames: {processed_features.shape}")
        
        return processed_features, batch_size

    def forward(self, dinov2_features, target_sentences=None, valid_frames=None):

        print("[START] : One forward pass of the model")

        # Process input dinov2_features
        dinov2_features, batch_size = self._process_dinov2_features(dinov2_features)

        # Linear Projection
        # print(f"Shape of dinov2_features before linear projection: {dinov2_features.shape}")
        dinov2_features = self.linear_adapter(dinov2_features)
        # print(f"Device of dinov2_features: {dinov2_features.device}")
        # print(f"Shape of dinov2_features after linear projection: {dinov2_features.shape}")
    
        # Input to Encoder: Extracted dinov2_features after pooling and linear projection
        encoder_outputs_from_dinov2_features = self.bart.model.encoder(inputs_embeds=dinov2_features)

        # encoder_outputs_from_dinov2_features = BaseModelOutput(last_hidden_state=dinov2_features)
        # print(f"[BART]: Shape of encoder_outputs_from_dinov2_features: {encoder_outputs_from_dinov2_features.shape}")

        if target_sentences is not None:
            # Prepare decoder inputs
            tokenized_target = self.tokenizer(target_sentences,
                                             return_tensors="pt",
                                             padding="max_length",
                                             truncation=True,
                                             max_length=MAX_TOKEN_LENGTH
                                             ).to(self.device)
            
            target_input_ids, target_attention_mask = tokenized_target.input_ids, tokenized_target.attention_mask

            # Mask out padding token losses
            target_input_ids[target_input_ids == self.tokenizer.pad_token_id] = -100
            print(f"[BART]: Shape of target_input_ids: {target_input_ids.shape}")
            # print(f"Value of target_input_ids : {target_input_ids}")

            print(f"[BART]: Shape of target Attention Mask: {target_attention_mask.shape}")

            visual_attention_mask = self._build_1D_attention_mask(valid_frames)
            print(f"[BART]: Shape of DINOv2 features Attention Mask: {visual_attention_mask.shape}")

            # Forward pass with teacher forcing using DINOv2 features output
            outputs = self.bart(encoder_outputs=encoder_outputs_from_dinov2_features,
                                                       attention_mask = visual_attention_mask,
                                                       labels=target_input_ids)    # model handles shifting of sentences internally

            print("[END] : One forward pass of the model")
            
            print(f"[BART]: Shape of the text output logits: {outputs.logits.shape}")

            # Using the logits, generate the corresponding text
            text_token_ids = torch.argmax(outputs.logits, dim=-1) # Get the token with the highest probability

            decoded_sentences = self.tokenizer.batch_decode(text_token_ids, 
                                                  skip_special_tokens=True, 
                                                  clean_up_tokenization_spaces=False)
            
           #  print(f"[BART]: Decoded sentences during training: {decoded_sentences}")

            return outputs.loss, outputs.logits, decoded_sentences

        else:
            # Inference
            generated_ids = self.bart.generate(
                encoder_outputs=encoder_outputs_from_dinov2_features,
                # attention_mask=full_attention_mask,
                max_length=MAX_TOKEN_LENGTH,
                num_beams=4,
                early_stopping=True
            )
            decoded_sentences = self.tokenizer.batch_decode(generated_ids, 
                                                  skip_special_tokens=True, 
                                                  clean_up_tokenization_spaces=False)
            print("[END] : One forward pass of the model for inference")
            return decoded_sentences
        
    def _build_1D_attention_mask(self, valid_frames):
        """
        Create attention mask for visual features
        Args:
            valid_frames: List of valid frame counts per video
        Returns:
            Attention mask tensor (1 for attending, 0 for ignoring)
            This mask is converted into an additive 2d attention mask in Llama
        """
        batch_size = len(valid_frames)
        attention_mask = torch.zeros((batch_size, MAX_FRAME_SIZE), device=self.device)
        for i, valid in enumerate(valid_frames):
            attention_mask[i, :valid] = 1
            
        return attention_mask
    

    def _build_2D_attention_mask(self, valid_frames, MAX_FRAME_SIZE, MAX_TOKEN_LENGTH):
        """
        Returns attention mask for a batch in the form of (B,480+90,480+90)
        """
        B= valid_frames.shape[0]
        total_len = MAX_FRAME_SIZE + MAX_TOKEN_LENGTH
        attention_mask = torch.zeros((B, total_len, total_len), device=self.device)

        # For each video, prepare attention mask individually:
        for b in range(B):
            v= valid_frames[b]

            # Valid frames attend to each other
            attention_mask[b, :v, :v] = 1

            # All text tokens attend to valid visual tokens
            attention_mask[b, MAX_FRAME_SIZE:, :v] = 1

            # Text tokens attend to previous text tokens (causal)
            text_start = MAX_FRAME_SIZE
            text_end = total_len
            causal = torch.tril(torch.ones((MAX_TOKEN_LENGTH, MAX_TOKEN_LENGTH), device=self.device))
            attention_mask[b, text_start:text_end, text_start:text_end] = causal

            # # Final attention mask
            # print('the shape of 1 attention mask: {}'.format(attention_mask.shape))
            # print('the 1 attention mask looks like: {}'.format(attention_mask))

        #print('the shape of final attention mask: {}'.format(attention_mask.shape))
        #print('the final attention mask looks like: {}'.format(attention_mask))
        return attention_mask


BART_FEATURE_DIMS = {
    "bart-base": 768,
    "bart-large": 1024
}