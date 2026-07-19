# EECE 5644 Final Project

## What this project does
This repository builds a fall detection classifier using the FallAllD dataset. This dataset is a collection of wearable sensor recordings capturing simulated falls and everyday activities (ADLs) across 15 subjects wearing accelerometer/gyroscope/magnetometer/barometer devices at the waist, wrist, and neck. This code converts thousands of raw sensor files into a single structured dataset, engineers statistical features per trial, and prepares train/validation/test splits for training a model that predicts whether a given recording is a fall or a normal activity.

## Repository contents
| File | Description |
|---|---|
| `convert_to_pkl.ipynb` | Parses all raw `.dat` sensor files, groups them by recording, and consolidates everything into a single `FallAllD.pkl` file |
| `detecting_activity.ipynb` | Loads `FallAllD.pkl`, cleans/validates the data, engineers features, and builds train/val/test splits for modeling |
| `ActivityID2Str.m` | Original MATLAB reference script (provided with the dataset) mapping each numeric ActivityID to its activity description |
| `FallAllD.pkl` | Consolidated dataset (one row per recording); available via Google Drive since it exceeds GitHub's file size limits (see step 2 below) |
| `requirements.txt` | Python library versions needed to reproduce this work |
| `README.md` | This file |

## Dataset
**FallAllD**: A Comprehensive Dataset of Human Falls and Activities of Daily Living, hosted on [IEEE DataPort](https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living) (Saleh, Abbas, and Le Bouquin Jeannès, *IEEE Sensors Journal*).

- **15 subjects**, each wearing 3 wearable data loggers (waist, wrist, neck)
- **6,605 trial recordings** (1,722 falls, 4,883 ADLs), reconstructed from 26,420 raw `.dat` files and there are 4 sensor files (Accelerometer, Gyroscope, Magnetometer, Barometer) per recording
- **79 activity types**: 35 distinct fall types (ActivityID 101–135) and 44 distinct ADL types (ActivityID 1–44)
- **Target variable**: binary: ADL = 0, Fall = 1

Raw filenames follow the pattern `S<SubjectID>_D<Device>_A<ActivityID>_T<TrialNo>_<SensorLetter>.dat` (e.g. `S01_D1_A013_T01_A.dat` = Subject 1, Device 1, Activity 13, Trial 1, Accelerometer).  `convert_to_pkl.ipynb` parses this filename convention directly and reconstructs the 8-field structure described in the dataset's documentation (`SubjectID`, `Device`, `ActivityID`, `TrialNo`, `Acc`, `Gyr`, `Mag`, `Bar`).

## How to run

1. Clone this repository.
2. Add the FallAllD data in the repo root. You have two options:
   - **Option A**: download the full raw dataset from [IEEE DataPort](https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living) (or from [Google Drive](https://drive.google.com/drive/folders/1HnCgKUxdF5X9yMeVMPyDVdQ2b4CNxmuD?usp=drive_link)) into a folder named `FallAllD`, then run `convert_to_pkl.ipynb` to regenerate `FallAllD.pkl` from scratch.
   - **Option B** (faster): skip the raw data and `convert_to_pkl.ipynb` entirely. Just download the pre-built `FallAllD.pkl` from [Google Drive](https://drive.google.com/drive/folders/1HnCgKUxdF5X9yMeVMPyDVdQ2b4CNxmuD?usp=drive_link) and place it in the repo root.
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Open `detecting_activity.ipynb` in Jupyter or VS Code. It loads `FallAllD.pkl`, checks for missing/invalid sensor values, encodes the fall/ADL target and device position, engineers per-trial statistical features (mean, median, std, max, min, RMS per sensor axis), and builds subject-wise train/validation/test splits (so no subject appears in more than one split).
5. Run all cells from top to bottom (**Restart Kernel + Run All**). The final cells output the cleaned, feature-engineered `X_train`/`X_val`/`X_test` matrices and corresponding labels, ready for model training.