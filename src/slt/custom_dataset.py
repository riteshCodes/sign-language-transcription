import os
import sys

import av
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(".")

from src.slt.configs import MAX_FRAME_SIZE


class FramesAndTextDataset(Dataset):
    def __init__(self,
                 video_root_dir,
                 csv_file_path,
                 frames_per_second=None,
                 transform=None):
        """ 
        Args:
            video_dir (str): Directory containing all video files
            csv_file_path (str): CSV file with video data information and corresponding target sentences
            transform(optional): Optional transform to be applied to frames
        """
        self.video_root_dir = video_root_dir

        unfiltered_dataframe = pd.read_csv(csv_file_path, delimiter='\t')
        print(f"[INFO] Unfiltered dataset contains {len(unfiltered_dataframe)} samples.")
        # Filter to only rows where the video file exists
        valid_rows = []
        for _, row in unfiltered_dataframe.iterrows():
            video_path = os.path.join(self.video_root_dir, row['SENTENCE_NAME'] + ".mp4")
            if os.path.exists(video_path):
                valid_rows.append(row)
        self.dataframe = pd.DataFrame(valid_rows)
        print(f"[INFO] Filtered dataset contains {len(self.dataframe)} valid samples.")

        self.frames_per_second = frames_per_second
        
        self.transform = transform

    def _extract_frames_tensor(self, video_path):
        """
        Extract frames from given video in the path
        Args:
            video_path (str): Path to the video file
        
        Returns:
            torch.Tensor: Equivalent tensor of given video frames
        """
        frames_tensor = []

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                # print(f"[INFO] Video:{video_path} is in FPS: {fps}")

                if self.frames_per_second is None or self.frames_per_second > fps: # If no fps is given or if the given fps is higher than the video's fps
                    frame_interval = 1  # Extract all frames
                else:
                    frame_interval = int(fps / self.frames_per_second)

                for frame_count, frame in enumerate(container.decode(stream)):
                    if frame_count % frame_interval == 0:
                        img = frame.to_image().convert('RGB')  # PIL Image

                        if self.transform is not None:
                            img = self.transform(img)  # Transformed Image

                        frames_tensor.append(img)

                    # Early stopping if enough frames are extracted (Truncation)
                    if len(frames_tensor) >= MAX_FRAME_SIZE:
                        # print(f"Maximum frame size reached during feature extraction, extracted frames: {len(frames_tensor)}")
                        break

        except Exception as exc:
            print(f"Error processing video {video_path} : {exc}")

        # print(f"[INFO] Total frames extracted: {len(frames_tensor)}")
        valid_frame_len = len(frames_tensor)
        # Stack frames into a single tensor
        stacked_frames_tensor = torch.stack(frames_tensor, dim=0)
        # Shape: [Actual_Frame_Size, Channel, Height, Width]

        # Pad or truncate the frames to a fixed size
        processed_frames_tensor = self._pad_frames(
            frames_tensor=stacked_frames_tensor,
            frame_size=MAX_FRAME_SIZE)

        # print(f"Total extracted frames' tensor size {processed_frames_tensor.shape}")

        return processed_frames_tensor, valid_frame_len

    def _pad_frames(self, frames_tensor, frame_size):
        """
        Pad frames to a fixed length
        Args:
            frames_tensor (torch.Tensor): Input frames tensor
        Returns:
            torch.Tensor: Padded frames tensor
        """
        num_frames, channels, height, width = frames_tensor.shape

        if num_frames < frame_size:
            # print(f"Padded frames to MAX_FRAME_SIZE:{MAX_FRAME_SIZE} from: {num_frames}")
            # Pad with zero tensors
            padding = frame_size - num_frames
            padded_tensor = torch.zeros(
                (padding, channels, height, width),
                dtype=frames_tensor.dtype
            )

            # Shape: [Normalized_Frame_Size, Channel, Height, Width]
            frames_tensor = torch.cat([frames_tensor, padded_tensor], dim=0)

        return frames_tensor

    def __len__(self):
        """
        Return the number of video-text pairs in the dataset
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset
        Args:
            idx (int): Index of the item
        Returns:
            tuple: (frames, target_sentence)
        """
        # Get target_sentence from corresponding csv file
        row = self.dataframe.iloc[idx]
        sentence_name = row['SENTENCE_NAME']
        target_sentence = row['SENTENCE']

        # Get frames of corresponding target_sentence video
        video_path = os.path.join(self.video_root_dir, sentence_name + ".mp4")

        # Extract frames
        frames_tensor, num_valid_frames = self._extract_frames_tensor(video_path)

        return (frames_tensor, target_sentence, num_valid_frames)


def collate_fn(batch):
    # batch_frames_tensor: [Batch, MAX_FRAME_SIZE, C, H, W]
    batch_frames_tensor = torch.stack([item[0] for item in batch], dim=0)
    batch_target_sentences = [item[1] for item in batch]
    batch_valid_frame_counts = [item[2] for item in batch]
    return batch_frames_tensor, batch_target_sentences , batch_valid_frame_counts