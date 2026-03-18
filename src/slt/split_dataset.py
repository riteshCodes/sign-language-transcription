import os
import random
import sys
from collections import defaultdict

import pandas as pd

sys.path.append(".")

from src.slt.configs import BASE_DATA_DIR, TEXT_REALIGNED_DIR, TRAIN_CSV_PATH


def split_csv_into_train_val(
    original_csv_path,
    train_csv_path,
    val_csv_path,
    seed=42
):
    random.seed(seed)
    desired_val_ratio = 0.2
    df = pd.read_csv(original_csv_path, delimiter='\t')
    print(f"[INFO] Original dataset has {len(df)} rows")

    # Group by VIDEO_ID (first column)
    grouped = defaultdict(list)
    for idx, row in df.iterrows():
        video_id = row[0]
        grouped[video_id].append(idx)

    train_indices = []
    val_indices = []

    for video_id, indices in grouped.items():
        count = len(indices)
        if count < 2:
            train_indices.extend(indices)
        else:
            # Decide how many samples to use for validation
            if count >= 20:
                val_count = 5
            elif count >= 10:
                val_count = 3
            elif count >= 5:
                val_count = 2
            else:
                val_count = 1

            # Randomly choose val_count indices for validation
            val_samples = random.sample(indices, val_count)
            train_samples = list(set(indices) - set(val_samples))

            val_indices.extend(val_samples)
            train_indices.extend(train_samples)

    # Adjust to maintain 80-20 split
    total_samples = len(train_indices) + len(val_indices)
    desired_val_size = int(desired_val_ratio * total_samples)

    print(
        f"[INFO] Pre-adjustment: Train = {len(train_indices)}, Val = {len(val_indices)}")
    current_val_size = len(val_indices)

    if current_val_size < desired_val_size:
        needed = desired_val_size - current_val_size
        move_indices = random.sample(train_indices, needed)
        for idx in move_indices:
            train_indices.remove(idx)
            val_indices.append(idx)
        print(
            f"[INFO] Moved {needed} samples from train → val to match 80-20 split.")
    elif current_val_size > desired_val_size:
        excess = current_val_size - desired_val_size
        move_indices = random.sample(val_indices, excess)
        for idx in move_indices:
            val_indices.remove(idx)
            train_indices.append(idx)
        print(
            f"[INFO] Moved {excess} samples from val → train to match 80-20 split.")

    # Final sanity check
    assert len(set(train_indices) & set(val_indices)
               ) == 0, "Train and val sets overlap!"

    # Save CSVs
    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)

    print(
        f"[INFO] Final: Train = {len(train_df)}, Val = {len(val_df)} ({len(val_df) / total_samples:.2%} of total)")

    train_df.to_csv(train_csv_path, sep='\t', index=False)
    val_df.to_csv(val_csv_path, sep='\t', index=False)
    print(f"[INFO] Saved train to {train_csv_path}, val to {val_csv_path}")


if __name__ == "__main__":
    train_csv_path = os.path.join(BASE_DATA_DIR, "train",
                                  TEXT_REALIGNED_DIR, "how2sign_realigned_train_subsampled.csv")
    val_csv_path = os.path.join(BASE_DATA_DIR, "val",
                                TEXT_REALIGNED_DIR, "how2sign_realigned_val_subsampled.csv")
    split_csv_into_train_val(original_csv_path=TRAIN_CSV_PATH,
                             train_csv_path=train_csv_path,
                             val_csv_path=val_csv_path)
