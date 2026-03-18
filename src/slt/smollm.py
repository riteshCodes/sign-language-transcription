import sys
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(".")

from src.slt.configs import (
    DEVICE, MAX_FRAME_SIZE, MAX_TOKEN_LENGTH, DINOv2_FEATURE_DIMS
)

class SLTmodelSmolLM(nn.Module):

    def __init__(self, dinov2_model_name="dinov2_vitl14", smollm_model_name="HuggingFaceTB/SmolLM2-360M"):
        super().__init__()

        self.device = DEVICE

        # SmolLM model
        self.smollm = AutoModelForCausalLM.from_pretrained(smollm_model_name,
                                                           torch_dtype=torch.bfloat16)
        self.smollm.to(self.device)
        # print(f"Device of SLTmodelSmolLM : {self.device}")

        # SmolLM tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(smollm_model_name)
        self._add_special_tokens() # Ensure special tokens are added to the model

        # Linear Projection Layer - use same dtype as SmolLM model
        self.decoder_hidden_size = self.smollm.config.hidden_size
        self.dinov2_hidden_size =  DINOv2_FEATURE_DIMS[dinov2_model_name]
        self.linear_adapter = nn.Linear(
            self.dinov2_hidden_size, self.decoder_hidden_size, dtype=torch.bfloat16)
        self.linear_adapter.to(self.device)

    def _process_dinov2_features(self, dinov2_features: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Process DINOv2 features to standardize their shape and dimensions.
        Args:
            dinov2_features: Input DINOv2 features
        Returns:
            Tuple of (processed_features, batch_size)
        """

        # print(f"[SmolLM]: Shape of input dinov2_features: {dinov2_features.shape}")
        # Convert to bfloat16 to match SmolLM model dtype
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
            #     f"[SmolLM]: Shape of input dinov2_features [CLS] after reshaping: {processed_features.shape}")

        else:  # Patch token features
            batch_and_frame, patch_dim, feat_dim = dinov2_features.shape  # [(batches*frames), patches, feature_dimension]
            batch_size = batch_and_frame // MAX_FRAME_SIZE
            processed_features = dinov2_features.view(
                batch_size, MAX_FRAME_SIZE, patch_dim, feat_dim
            )
            # Mean Pool over patches and collapse dimension -> Returns shape of (seq_length, MAX_FRAME_SIZE, feat_dim)
            processed_features = processed_features.mean(dim=2)
            # print(
            #     f"[SmolLM]: Shape of input dinov2_features [Patch] after pooling across frames: {processed_features.shape}")
        
        return processed_features, batch_size
    
    def _build_2D_attention_mask(self, valid_frames, MAX_FRAME_SIZE, token_attention_mask=None, num_text_tokens=None):
        """
        Returns attention mask for a batch in the form of (Batch, MAX_FRAME_SIZE+MAX_TOKEN_LENGTH, MAX_FRAME_SIZE+MAX_TOKEN_LENGTH)
        Args:
            valid_frames: List of valid frame counts per video
            MAX_FRAME_SIZE: Max frames per video
            token_attention_mask: attention mask returned from the tokenized target sentence during training
            num_text_tokens: number of generated text tokens during generation
        Info about attention mask:
        1. All visual features attend each other
        2. All generated tokens attend all visual features
        3. Generated tokens attend previously generated tokens (causal mask)
        Values: 0.0 for attention and -inf for ignoring (additive mask)
        """
        # print(f"[SmolLM]: Given valid_frames in the batch: {valid_frames}")

        batch_size = len(valid_frames)
        if token_attention_mask is not None:
            MAX_TOKEN_LENGTH = token_attention_mask.shape[1]
            total_len = MAX_FRAME_SIZE + MAX_TOKEN_LENGTH
        elif num_text_tokens is not None:
            total_len = MAX_FRAME_SIZE + num_text_tokens
        else:
            raise ValueError("You must provide either token_attention_mask or num_text_tokens.")

        ignore_value = torch.tensor(torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16, device=self.device)
        attention_mask = torch.full((batch_size, total_len, total_len),fill_value=ignore_value.item(),dtype=torch.bfloat16,device=self.device)

        # For each video, prepare attention mask individually:
        # for b in range(batch_size):
        for b, v in enumerate(valid_frames):  # v: number of valid frames per video in the batch

            # Valid frames attend to each other
            attention_mask[b, :v, :v] = 0.0

            # Get number of valid (non-padded) text tokens
            if token_attention_mask is not None:
                num_valid_tokens = token_attention_mask[b].sum().item()  # used during training (so that we can ignore the padded target tokens)
            else:
                num_valid_tokens = num_text_tokens  # already BOS + generated (used during generation)

            # All text tokens attend to valid visual tokens
            attention_mask[b, MAX_FRAME_SIZE:MAX_FRAME_SIZE + num_valid_tokens, :v] = 0.0

            # Text tokens attend to previous text tokens (causally)
            if num_valid_tokens > 0:
                # Causal mask for valid text tokens only
                causal_mask = torch.tril(torch.ones((num_valid_tokens, num_valid_tokens), dtype=torch.bool, device=self.device))
                attention_mask[b, MAX_FRAME_SIZE:MAX_FRAME_SIZE + num_valid_tokens, MAX_FRAME_SIZE:MAX_FRAME_SIZE + num_valid_tokens].masked_fill_(causal_mask, 0.0)

        return attention_mask


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

    def _add_special_tokens(self):

        # self.tokenizer.padding_side = "left" # Right is the default value in SmolLM
        # Info:
        # SmolLM2 uses a GPT2Tokenizer 
        # with a vocabulary size of 49,152 tokens and a maximum sequence length of 8,192 tokens
        # Byte-Pair Encoding (BPE) tokenizer
        # GPT2Tokenizer -> BOS token == EOS token (default)
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

        self.tokenizer.padding_side = "right" # Right is the default value in SmolLM

        print("BOS token:", self.tokenizer.bos_token, "ID:", self.tokenizer.bos_token_id)
        print("EOS token:", self.tokenizer.eos_token, "ID:", self.tokenizer.eos_token_id)
        print("PAD token:", self.tokenizer.pad_token, "ID:", self.tokenizer.pad_token_id)


    def forward(self, dinov2_features, target_sentences=None, valid_frames=None):

        # self.smollm.train()
        
        # Process input dinov2_features
        dinov2_features, batch_size = self._process_dinov2_features(dinov2_features)
        
        # Projection to LM hidden size
        dinov2_features = self.linear_adapter(dinov2_features)
        # print(
        #     f"[SmolLM]: Shape of dinov2_features after linear projection: {dinov2_features.shape}")

        # tokenize target, it will be added to the the input sequence (after the features)
        if target_sentences is not None:
            # Adding EOS and BOS token explicitly to the target sentences
            target_sentences = [f"{self.tokenizer.bos_token} {s} {self.tokenizer.eos_token}" for s in target_sentences]

            tokenized = self.tokenizer(target_sentences,
                                       return_tensors="pt",
                                       padding="max_length",
                                       truncation=True,
                                       max_length=MAX_TOKEN_LENGTH,
                                       add_special_tokens=True
                                       ).to(self.device)

            # print(f"[SmolLM]: Value of tokenized input_ids: {tokenized.input_ids}")

            # print(f"[SmolLM]: Shape of tokenized input_ids: {tokenized.input_ids.shape}")

            decoder_inputs = self.smollm.model.embed_tokens(tokenized.input_ids)
            
            # print(f"[SmolLM]: Shape of decoder_inputs from embedding layer: {decoder_inputs.shape}")

            # print(f"Device of decoder_inputs: {decoder_inputs.device}")

            # Concatenate video frames' dinov2_features and target sentence's decoder_inputs
            # Along sequence dimension (dim=1)
            # [visual_features] + [BOS] + [text tokens] + [EOS]
            inputs_embeds = torch.cat([dinov2_features, decoder_inputs], dim=1)
            # print(f"[SmolLM]: Shape of inputs_embeds to the model: {inputs_embeds.shape}")

            # print(f"Device of inputs_embeds: {inputs_embeds.device}")

            #NOTE: Use this for 1D attention mask:

            #Attention mask should match the shape and order of the concatenated input embeddings: [batch_size, sequence_length] -> (input_frames_batch_size, MAX_FRAME_SIZE)
            #Attention mask for extracted DINOv2 features
            # feature_attention_mask = self._build_1D_attention_mask(valid_frames)
            # print(f"[SmolLM]: Shape of DINOv2 feature Attention Mask: {feature_attention_mask.shape}")
            # full_attention_mask = torch.cat([feature_attention_mask, tokenized.attention_mask], dim=1)
            # #Shape: The mask should be [batch_size, total_sequence_length=adding both video features and text tokens]
            # #The mask must match the order of concatenated embeddings: [video features | text tokens]
            # print(f"[SmolLM]: Shape of full_attention_mask: {full_attention_mask.shape}")
            # token_attention_mask = tokenized['attention_mask']

            # To make sure the model does not cheat, it needs an attention mask that will prevent it to access the tokens after token i when trying to predict the token i+1 in the sentence

            #NOTE: Use this for 2D attention mask:

            # Custom 2D attention mask for each video+target_sentence in batch (Overall: 3D mask (batch_size, 2D matrix)
            token_attention_mask = tokenized['attention_mask']  # need this in order to ignore the padded tokens for each sentence
            full_attention_mask = self._build_2D_attention_mask(valid_frames= valid_frames, 
                                                                MAX_FRAME_SIZE=MAX_FRAME_SIZE, 
                                                                token_attention_mask = token_attention_mask)
            # Make it compatible with causal mask in pre-trained llama
            full_attention_mask = full_attention_mask[:, None, :, :]



            # Labels to the model

            # Assign label = -100 (ignore for loss computation) for padded text tokens
            text_labels = tokenized.input_ids.clone()
            text_labels[text_labels == self.tokenizer.pad_token_id] = -100
            # print(f"[SmolLM]: Shape of text_labels: {text_labels.shape}")
            # print(f"[SmolLM]: Device of text labels: {text_labels.device}")
            # Assign label = -100 (ignore for loss computation) for all extracted visual tokens
            visual_labels = torch.full((batch_size, MAX_FRAME_SIZE), -100).to(text_labels.device)
            # print(f"[SmolLM]: Shape of visual_labels: {visual_labels.shape}")

            # Concatenate visual labels + text labels
            labels = torch.cat([visual_labels, text_labels], dim=1)
            # print(f"[SmolLM]: Shape of labels (visual features and tokenized input_ids): {labels.shape}")
            
            # Info:
            # For causal language modeling:
            # Manually shifting the labels when using HuggingFace's transformers library not required
            # Model's loss function (cross-entropy) is implemented to automatically shift the labels internally (shifts to the left)
            # Loss is computed between the prediction at position i and the label at position i+1
            # Causal language modeling: the model has to predict the next token in the sentence (so the labels are the same as the inputs shifted to the right).


            # Forward Pass to SmolLM
            outputs = self.smollm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=labels 
            )

            # print(f"[SmolLM]: Shape of the full output logits: {outputs.logits.shape}")

            text_logits = outputs.logits[:, MAX_FRAME_SIZE:, :].contiguous()
            # print(f"[SmolLM]: Shape of the text output logits: {text_logits.shape}")

            # Using the logits, generate the corresponding text
            token_ids = torch.argmax(text_logits, dim=-1) # Get the token with the highest probability
            # print(f"[SmolLM]: Shape of token_ids (after getting tokens greedily): {token_ids.shape}")
            
            # Decode the token ids to text - decode all sequences in the batch
            # Mask out padded tokens before decoding
            generated_text = []
            for i in range(batch_size):
                valid_len = token_attention_mask[i].sum().item()  # since default attention mask ignores the padded tokens
                # Only consider the valid text tokens after visual tokens, excluding the padded target text tokens (<visual_tokens> + <valid_tokens> + <padded_tokens>)
                # This ensures that our generation logic is consistent during train and eval
                valid_token_ids = token_ids[i][:valid_len].tolist()

                # Stop at EOS if present
                if self.tokenizer.eos_token_id in valid_token_ids:
                    valid_token_ids = valid_token_ids[:valid_token_ids.index(self.tokenizer.eos_token_id)]

                decoded = self.tokenizer.decode(valid_token_ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True)
                generated_text.append(decoded)
            # print(f"[SmolLM]: List of generated sentence size: {len(generated_text)}")
            # print(f"[SmolLM]: Generated text: {generated_text}")

            return outputs.loss, outputs.logits, generated_text

        else:
            generated = self.generate(dinov2_features, batch_size, valid_frames, MAX_TOKEN_LENGTH, 5)
            return generated


    @torch.no_grad()
    def generate(self, dinov2_features, batch_size, valid_frames, MAX_TOKEN_LENGTH, repetition_threshold):

        self.smollm.eval()

        bos_token_id = self.tokenizer.bos_token_id
        bos_embed = self.smollm.get_input_embeddings()(torch.tensor([bos_token_id], device=self.device)).unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 1, D)

        # Init sequence: visual features + BOS
        inputs_embeds = torch.cat([dinov2_features, bos_embed], dim=1)

        # #NOTE: use this for 1d attention mask:
        # feature_attention_mask = self._build_1D_attention_mask(valid_frames)
        # full_attention_mask = torch.cat([feature_attention_mask,torch.ones((batch_size, 1), device=self.device)], dim=1)  # visual + <bos>


        # NOTE: use this for 2d attention mask
        full_attention_mask = self._build_2D_attention_mask(
            valid_frames=valid_frames,
            MAX_FRAME_SIZE=dinov2_features.shape[1],
            num_text_tokens=1,  #bos token
        )
        full_attention_mask = full_attention_mask[:, None, :, :]


        # print(f'[SmolLM]: Shape of initialized input embeds in Generation (with added BOS token): {inputs_embeds.shape}')
        # print(f'[SmolLM]: Shape of initialized attention mask in Generation (with added BOS token): {full_attention_mask.shape}')


        # Init generated token IDs
        generated_token_ids = [[] for _ in range(batch_size)]
        done = [False] * batch_size
        # Repeat generation loop
        for j in range(MAX_TOKEN_LENGTH):
            outputs = self.smollm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask
            )

            next_token_logits = outputs.logits[:, -1, :]  # (B, V)
            next_token_ids = torch.argmax(next_token_logits, dim=-1)  # Greedy decode (B,)

            # Append to generated_token_ids
            for i, token_id in enumerate(next_token_ids.tolist()):
                if not done[i]:
                    generated_token_ids[i].append(token_id)

            # Check stopping conditions
            for i in range(batch_size):
                if done[i]:
                    continue
                tokens = generated_token_ids[i]
                # Stop if EOS
                if tokens and tokens[-1] == self.tokenizer.eos_token_id:
                    done[i] = True
                # Stop if repeated too many times
                if len(tokens) >= repetition_threshold and len(set(tokens[-repetition_threshold:])) == 1:
                    done[i] = True

            # If all are done, stop early
            if all(done):
                break

            # Get embeddings of predicted tokens
            new_token_embeds = self.smollm.get_input_embeddings()(
                next_token_ids
            ).unsqueeze(1)  # (B, 1, D)

            # Append the new token to current inputs
            inputs_embeds = torch.cat([inputs_embeds, new_token_embeds], dim=1)

           ## NOTE: Use this for 1D attention mask:
           # # attend the new generated token
            # full_attention_mask = torch.cat([full_attention_mask,torch.ones((batch_size, 1), device=self.device)], dim=1)

            # NOTE: use this for 2D attention mask:
            # Rebuild full attention mask with updated token count
            full_attention_mask = self._build_2D_attention_mask(
                valid_frames=valid_frames,
                MAX_FRAME_SIZE=dinov2_features.shape[1],
                num_text_tokens=(j+1) + 1,  # 1 for BOS + (j+1) generated tokens (loop count)
            )
            full_attention_mask = full_attention_mask[:, None, :, :]


        # Decode sequences
        generated_text = []
        for ids in generated_token_ids:
            if self.tokenizer.eos_token_id in ids:
                ids = ids[:ids.index(self.tokenizer.eos_token_id)]
            text = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            generated_text.append(text)
            
        return generated_text
