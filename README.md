# Sign Language Transcription

## Description
This project focuses on developing a vision-to-text model that combines a vision transformer with a language model to transcribe sign language from video input. The model is designed to preprocess video data into uniform frames which are used as input for the vision transformer. The output from the vision transformer is then fed into a language model to generate transcriptions of the sign language. The project is based on American Sign Language (ASL) and utilizes the How2Sign dataset, which contains videos of ASL along with their corresponding English translations. The project leverages gloss-free approach, i.e., it does not rely on intermediate representations of sign language (glosses) but directly transcribes the video input into text.

## Preparing Dataset
- **`How2Sign`** dataset consists of the following modalities:
    1. `rgb_front_videos` : Green Screen RGB videos - Frontal View
    2. `rgb_side_videos` : Green Screen RGB videos - Side View
    3. `rgb_front_clips` : Green Screen RGB clips -- Frontal view
    4. `rgb_side_clips` : Green Screen RGB clips -- Side view
    5. `rgb_front_2D_keypoints` : B-F-H 2D Keypoints clips -- Frontal view 
    6. `english_translation` : English Translation
    6. `english_translation_re-aligned` : English Translation re-aligned  

The Green Screen RGB clips (`rgb_front_clips` and `rgb_side_clips`) were segmented using the original timestamps from the How2 dataset. Each clip corresponds to one sentence of the English translation. Note that this may not have a perfect alignment between the ASL video and the English translation due to the differences between both languages. The manual re-aligned clips can be obtained by segmenting the *Green Screen RGB videos* with the timestamps available in the *English Translation (manually re-aligned)* file.

**Using the `download_how2sign.sh` Script:**  
The provided script, `download_how2sign.sh` in the `/data` folder, automates the process of creating the necessary folder structure and downloading your desired modalities of the How2Sign dataset.

**To download specific modalities:**  
To use this script from the `/data` folder, first choose the modalities to download and pass them as arguments to the command running the script.

**Example:**
To download the *rgb_front_videos*, the *rgb_side_videos* and the *english_translation_re-aligned*, use the following command:

```bash
./download_how2sign.sh rgb_front_videos rgb_side_videos english_translation_re-aligned
```

## Installation
To install all the dependencies of the project, use the `pyproject.toml` file.

---

*Note: More detailed information about the implementation is given in the `implementation-overview.pdf`*