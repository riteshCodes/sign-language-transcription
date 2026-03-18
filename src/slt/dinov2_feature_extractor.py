import sys

import torch

sys.path.append(".")

from src.slt.configs import DEVICE, MAX_FRAME_SIZE


class DinoV2FeatureExtractor:

    """
    A feature extractor that processes input frames (images) using DINOv2 model
    """

    def __init__(self, model_name="dinov2_vitl14", with_cls_token=False, device=DEVICE):
        """
        Initialize the DinoV2 (default: dinov2_vitl14) feature extractor.
        Args:
            model_name: Name of the DinoV2 model to use
                        Options: "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"
        """

        # Load DINOv2 model
        print(f"LOADING {model_name} MODEL ...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        
        # Freeze DINOv2 model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Use DINOv2 in evaluation mode
        self.model.eval()

        self.model = self.model.to(device)
        self.device = device

        self.return_cls_token = with_cls_token

        print(f"MODEL {model_name} LOADED.")
        # print(f"Device of DinoV2FeatureExtractor : {self.device}")

    def extract_features(self, input_frames):
        """
        Extracts CLS token and patch token features from a batch of input frames using the DINOv2 model.
        Processes the input frames in batches and extracts two types of features for each frame:
            - `x_norm_clstoken`: The final hidden state of the CLS token after layer normalization  
            - `x_norm_patchtokens`: The final hidden states of the patch tokens after layer normalization

        Args:  
        input_frames (torch.Tensor): A tensor of shape (num_frames, channels, height, width) containing the input frames (images) to extract features from.

        Returns:  
        Tuple[torch.Tensor, torch.Tensor]:
            - all_cls_tokens (torch.Tensor): Concatenated CLS token embeddings of shape (num_frames, embedding_dim).
            - all_patch_tokens (torch.Tensor): Concatenated patch token embeddings of shape (num_frames, num_patches, embedding_dim).

        Notes:  
            - The frames are processed in chunks of size determined by MAX_FRAME_SIZE
        """

        cls_tokens_list = []
        patch_tokens_list = []

        batch_dim , num_frames, channels, height, width = input_frames.shape
        # print(f"[DINOv2]: Shape of input frames to DINOv2 feature extractor: {input_frames.shape}")

        # patch_size = height // 14    # DINOv2 config: height and width normalized and equal, using 14*14 patches
        # num_patches = patch_size * patch_size

        # Flatten across all videos and frames
        input_frames = input_frames.view(batch_dim * num_frames, channels, height, width)

        # Process frames in chunks
        # chunk_size = int(MAX_FRAME_SIZE / 30)
        chunk_size = MAX_FRAME_SIZE # Process frames in chunks of batches
        # print(f"Start input frames feature extraction in batches")
        for i in range(0, input_frames.size(0), chunk_size):
            batch_frames_chunk = input_frames[i:i+chunk_size].clone()
            # print(f"Shape of batch frames: {batch_frames_chunk.shape}")
            batch_frames_chunk = batch_frames_chunk.to(self.device)
            # print(f"Batch frame size: {batch_frames.shape}")

            # Extract features through DINOv2
            with torch.no_grad():
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs = self.model.forward_features(batch_frames_chunk)

                    cls_token = outputs["x_norm_clstoken"]
                    # print(f"Shape of single cls token: {cls_token.shape}")
                    cls_tokens_list.append(cls_token)

                    patch_tokens = outputs["x_norm_patchtokens"]
                    # print(f"Shape of single patch token: {patch_tokens.shape}")
                    patch_tokens_list.append(patch_tokens)

        # Concatenate all batches
        all_cls_tokens = torch.cat(cls_tokens_list, dim=0).to(self.device)

        all_patch_tokens = torch.cat(patch_tokens_list, dim=0).to(self.device)

        # Unflatten to [batch_size, num_frames, embedding_dim]
        # unflattened_cls_tokens = all_cls_tokens.view(batch_dim, num_frames, -1)
        # print(f"Shape of unflattened cls tokens: {unflattened_cls_tokens.shape}")
        # unflattened_patch_tokens = all_patch_tokens.view(batch_dim, num_frames, num_patches, -1)
        # print(f"Shape of unflattened patch tokens: {unflattened_patch_tokens.shape}")

        if self.return_cls_token:
            print(f"[DINOv2]: Shape of all extracted cls tokens: {all_cls_tokens.shape}")
            return all_cls_tokens
        else:
            print(f"[DINOv2]: Shape of all extracted patch tokens: {all_patch_tokens.shape}")
            return all_patch_tokens


# Output model feature dimensions
MODEL_FEATURE_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536
}