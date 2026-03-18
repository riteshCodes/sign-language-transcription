import io
import os
import sys

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(".")

from src.baseline.configs import (DEVICE, FRAMES_PER_SECOND, MAX_FRAME_SIZE,
                                  TRAIN_CSV_PATH, VIDEO_DIR)
from src.baseline.transforms import train_transform


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    """
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    # moves the decoding position to the start (timestamp 0) of the video stream
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def extract_frames_with_av(video_path, output_dir, image_format="jpg"):
    """
    Extracts frames from a video using PyAV and saves them as image files.
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        image_format (str): Image format to save frames ('jpg', 'png', etc.).
    Returns:
        int: Number of frames extracted and saved.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    total_frames = container.streams.video[0].frames
    print(f"total frames {total_frames}")

    # Get FPS (Frames Per Second)
    fps = float(video_stream.average_rate)
    print(f"Video FPS: {fps}")

    frame_count = 0
    key_frame_count = 0

    for frame in container.decode(video=0):
        img = frame.to_image()  # Convert to PIL.Image
        output_path = os.path.join(
            output_dir, f"frame_{frame_count:04d}.{image_format}"
        )
        img.save(output_path)
        frame_count += 1
        # Extract only key frames
        if frame.key_frame:
            key_frame_count += 1
            # Save the frame as an image (e.g., PNG)
            frame_path = os.path.join(
                output_dir, f"key_frame_{frame_count:04d}.{image_format}"
            )
            frame.to_image().save(frame_path)
            print(f"Total keyframes {key_frame_count}")
            print(f"Saved {frame_path}")

    container.close()
    print(f"Extraction complete: {frame_count} frames saved to '{output_dir}'.")
    return frame_count


def extract_videos_paths(batch_size=None):
    folder_path = os.path.join("data", "processed_frames")

    all_videos_paths = [
        os.path.join(folder_path, video_folder)
        for video_folder in os.listdir(folder_path)
    ]

    if batch_size is not None:
        all_videos_paths = all_videos_paths[:batch_size]
    return all_videos_paths


def extract_frame_paths(video_folder_path: str, batch_size=None):
    frames_paths = [
        os.path.join(video_folder_path, frame_path)
        for frame_path in os.listdir(video_folder_path)
        if frame_path.endswith((".png", ".jpg", ".jpeg"))
    ]

    if batch_size is not None:
        frames_paths = frames_paths[:batch_size]

    return frames_paths


def extract_resized_frames_from_video(video_path, output_dir):
    # Open the video file
    container = av.open(video_path)

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare a list to collect frame data
    data = []

    for frame_number, frame in enumerate(container.decode(video=0)):
        # Get timestamp in seconds
        timestamp = float(frame.pts * frame.time_base)

        # Convert frame to a PIL Image
        img = frame.to_image()

        # Resize image to 224x224 for efficiency (also DINOv2 input size)
        img_resized = img.resize((224, 224))

        # Optionally, store image in bytes (e.g., for keeping in DataFrame)
        with io.BytesIO() as output:
            img_resized.save(output, format="PNG")
            image_bytes = output.getvalue()

        # Save the image to disk
        img_resized.save(os.path.join(output_dir, f"frame_{frame_number:05d}.png"))

        # Append frame data
        data.append(
            {"frame_number": frame_number, "timestamp": timestamp, "image": image_bytes}
        )

    # Create a DataFrame to hold metadata
    df = pd.DataFrame(data)
    df.to_csv(
        os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0]),
        index=False,
        sep=";",
        encoding="utf-8",
    )

    # return df


def load_resized_frames_from_folder(folder_path, extensions={".png", ".jpg", ".jpeg"}):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[-1].lower()
        if ext in extensions:
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("RGB")  # Ensure RGB format
            images.append(img)
            filenames.append(filename)
    return images, filenames


def extract_frames_tensor(video_path, output_dir, frames_per_second):
    print(f"Input FPS: {frames_per_second}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            print(f"Video:{video_path} is in FPS: {fps}")

            if (
                frames_per_second is None or frames_per_second > fps
            ):  # If no fps is given or if the given fps is higher than the video's fps
                frame_interval = 1  # Extract all frames
            else:
                # frame_interval = min(fps, frames_per_second)
                frame_interval = int(fps / frames_per_second)

            print(f"Extraction in Frame Interval:{frame_interval}")

            total_frames = 0

            for frame_count, frame in enumerate(container.decode(stream)):
                if frame_count % frame_interval == 0:
                    print(f"Extracted Frame:{frame_count}")
                    img = frame.to_image().convert("RGB")  # PIL Image
                    output_path = os.path.join(
                        output_dir, f"frame_{frame_count:04d}.png"
                    )
                    img.save(output_path)
                    total_frames += 1

    except Exception as exc:
        print(f"Error processing video {video_path} : {exc}")

    print(f"Total frames extracted: {total_frames}")


def _extract_frames(video_path, output_dir, frames_per_second):
    print(f"Input FPS: {frames_per_second}")

    frames_tensor = []

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            duration = float(stream.duration * stream.time_base)
            print(f"Video:{video_path} is in FPS: {fps}")

            # Calculate target frame timestamps
            if frames_per_second is None:
                # If no FPS specified, use video's native FPS
                target_fps = fps
            else:
                target_fps = min(frames_per_second, fps)

            # Calculate exact timestamps for frame extraction
            frame_timestamps = []
            current_time = 0
            while current_time < duration:
                frame_timestamps.append(current_time)
                current_time += 1.0 / target_fps

            total_frames = 0
            frame_count = 0
            # Extract frames at precise timestamps
            for timestamp in frame_timestamps:
                # Seek to the closest timestamp
                print(f"Time stamp {timestamp}")
                print(f"Stream time base {stream.time_base}")
                container.seek(int(timestamp * stream.time_base))

                for frame in container.decode(stream):
                    frame_count += 1
                    print(f"Extracted Frame:{frame_count}")
                    # Convert frame to image
                    img = frame.to_image().convert("RGB")

                    if img is None:
                        continue

                    output_path = os.path.join(
                        output_dir, f"frame_{frame_count:04d}.png"
                    )
                    img.save(output_path)
                    total_frames += 1

                    frames_tensor.append(img)

                    break

                # Validate the actual FPS achieved
            actual_fps = len(frames_tensor) / duration
            if abs(actual_fps - target_fps) > 0.1:  # Allow 0.1 FPS deviation
                print(
                    f"Warning: Actual FPS ({actual_fps:.2f}) differs from target FPS ({target_fps:.2f})"
                )

            if len(frames_tensor) == 0:
                raise ValueError(
                    f"No frames could be extracted from video: {video_path}"
                )

    except Exception as exc:
        print(f"Error processing video {video_path}: {exc}")

    print(f"Total frames extracted: {total_frames}")


def pad_frames(frames_tensor, frame_size):
    """
    Pad given frames_tensor to a fixed length
    Args:
        frames_tensor (torch.Tensor): Input frames tensor
        frame_size (int): Desired frame size
    Returns:
        torch.Tensor: Padded frames tensor
    """
    num_frames, channels, height, width = frames_tensor.shape
    if num_frames < frame_size:
        padding = frame_size - num_frames
        padded_tensor = torch.zeros(
            (padding, channels, height, width), dtype=frames_tensor.dtype
        )
        frames_tensor = torch.cat([frames_tensor, padded_tensor], dim=0)
    elif num_frames > frame_size:
        frames_tensor = frames_tensor[:frame_size]
    return frames_tensor.half() # torch.float16


def store_processed_dataset(
    video_dir, csv_file_path, output_path, frames_per_second=None, transform=None):

    df = pd.read_csv(csv_file_path, delimiter="\t")
    print(f"[INFO] Unfiltered dataset contains {len(df)} samples.")
    # Filter to only rows where the video file exists
    valid_rows = []
    for _, row in df.iterrows():
        video_path = os.path.join(video_dir, row["SENTENCE_NAME"] + ".mp4")
        if os.path.exists(video_path):
            valid_rows.append(row)
    dataframe = pd.DataFrame(valid_rows)
    print(f"[INFO] Filtered dataset contains {len(dataframe)} valid samples.")

    data = []
    for _, row in dataframe.iterrows():
        video_name = row["SENTENCE_NAME"]
        video_path = os.path.join(video_dir, video_name + ".mp4")
        
        frames_tensor = []
        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                print(f"[INFO] Video:{video_path} is in FPS: {fps}")
                if frames_per_second is None or frames_per_second > fps:
                    frame_interval = 1
                else:
                    frame_interval = int(fps / frames_per_second)
                print(f"[INFO] Extracting frames from video: {video_path}")
                for frame_count, frame in enumerate(container.decode(stream)):
                    if frame_count % frame_interval == 0:
                        img = frame.to_image().convert("RGB")
                        if transform is not None:
                            img = transform(img)
                        else:
                            img = transforms.ToTensor()(img).half() # torch.float16 
                        frames_tensor.append(img)
                        if len(frames_tensor) >= MAX_FRAME_SIZE:
                            break
            if len(frames_tensor) == 0:
                print(f"Warning: No frames extracted for {video_path}")
                continue
            print(f"[INFO] Completed Extraction of frames from video: {video_path}")
            valid_frames = len(frames_tensor)
            stacked = torch.stack(frames_tensor, dim=0)
            padded_frames = pad_frames(stacked, MAX_FRAME_SIZE)
            data.append(
                {
                    "video_name": video_name,
                    "frames_tensor": padded_frames,
                    "valid_frames_count": valid_frames,
                    # "target_sentence": sentence,
                }
            )
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    torch.save(data, output_path)
    print(f"Data stored in {output_path}")


def load_processed_dataset(processed_data_path, index):
    data = torch.load(processed_data_path, map_location=DEVICE)
    item = data[index]
    return item["video_name"], item["frames_tensor"], item["valid_frames_count"]


if __name__ == "__main__":
    processed_dataset = f"processed_dataset_FPS_{FRAMES_PER_SECOND}_frames_size_{MAX_FRAME_SIZE}.pth"
    processed_dataset_path = os.path.join(VIDEO_DIR, processed_dataset)
    video_dir = os.path.join(VIDEO_DIR, "Test")
    # processed_dataset_path = os.path.join(TRAIN_VIDEO_DIR, "Test", processed_dataset)
    # video_dir = os.path.join(TRAIN_VIDEO_DIR, "Test")
    store_processed_dataset(
        video_dir=video_dir, 
        csv_file_path=TRAIN_CSV_PATH, 
        output_path=processed_dataset_path, 
        frames_per_second=FRAMES_PER_SECOND,
        transform=train_transform
    )
    # video_name, frames_tensor, valid_frames_count = load_processed_dataset(processed_dataset_path, 0)
    # print(f"Video: {video_name}")
    # print(f"Frames extracted: {frames_tensor}")
    # print(f"Valid frames count: {valid_frames_count}")
