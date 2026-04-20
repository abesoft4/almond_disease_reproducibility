# Almond Disease Reproducibility Package

This package provides the exact grouped split plan, the best-performing EfficientNetV2-B0 checkpoint, and standalone evaluation scripts for calibration and explainability analysis. It is designed for reproducibility of the manuscript without requiring training from scratch.

## What Is Included

- `model/best.keras`: best-performing trained checkpoint.
- `model/classes.json`: class-name ordering used by the checkpoint.
- `splits/grouped_split_plan.json`: grouped split definition used in the study.
- `scripts/load_inference_artifacts.py`: shared loader for model, class map, preprocessing, and temperature scaling.
- `scripts/analyze_image.py`: single-image inference, calibrated probabilities, Grad-CAM, and Integrated Gradients.
- `scripts/analyze_folder.py`: batch version of the same analysis for a directory of images.
- `scripts/check_environment.py`: prints Python, library versions, TensorFlow device visibility, and metadata.
- `metadata/model_manifest.json`: best-model metadata and calibration settings.
- `metadata/preprocessing.json`: preprocessing and augmentation details inferred from the manuscript workflow.
- `metadata/environment.json`: reproducibility environment and hardware notes.
- `requirements.txt`: runtime dependencies for package evaluation.

## Scientific Scope

This package reproduces the manuscript's evaluation workflow around the selected EfficientNetV2-B0 model. In particular, it supports:

- temperature-scaled probability analysis,
- explainability analysis with Grad-CAM and Integrated Gradients,
- reuse of the fixed grouped split plan,
- inspection of preprocessing and runtime environment details.

It does not retrain models. The purpose is direct verification of the published checkpoint and post-hoc analyses.

## Grouped Validation Note

The manuscript describes grouped plant-level validation operationalized through duplicate and near-duplicate grouping before fold assignment. The split file included here stores the exact image lists used per fold. It therefore reproduces the evaluation protocol used in the manuscript, even though the saved split file itself is expressed as file lists rather than as a separate plant-ID table.

## Preprocessing Used For Inference

The exported scripts apply the same inference preprocessing used by the selected manuscript model:

- load RGB image,
- resize to `224 x 224`,
- convert to `float32`,
- scale to `[0, 1]`,
- apply `tensorflow.keras.applications.efficientnet_v2.preprocess_input(x * 255.0)`.

Training-time augmentation settings are documented in `metadata/preprocessing.json` for completeness, but they are not applied during evaluation.

## Environment

Validated with:

- Python `3.12.10`
- TensorFlow `2.21.0`
- Keras `3.14.0`
- NumPy `2.4.4`
- h5py `3.14.0`
- Matplotlib `3.10.8`
- Pillow `12.2.0`

The model training associated with this study was run on a Dell Precision 5820 Tower workstation with an Intel Xeon W-2155 CPU, 32 GB RAM, and an NVIDIA Quadro P4000 GPU with 8 GB VRAM. The packaged evaluation scripts were validated in the current project environment on Windows 10 Pro 64-bit.

## Installation

From the package root:

```bash
pip install -r requirements.txt
```

## Single-Image Analysis

```bash
python scripts/analyze_image.py path/to/image.jpg
```

This writes:

- a JSON file with raw and temperature-scaled probabilities,
- a PNG panel with the input image, Grad-CAM overlay, and Integrated Gradients heatmap.

Default output location:

- `outputs/single_image/`

## Folder Analysis

```bash
python scripts/analyze_folder.py path/to/folder
```

This writes:

- one JSON file per image,
- one explainability PNG per image,
- `batch_summary.json`,
- `batch_summary.csv`.

Default output location:

- `outputs/batch/`

## Environment Check

```bash
python scripts/check_environment.py
```

This prints the runtime environment, TensorFlow version, Keras version, visible devices, and the package metadata needed for reproducibility reporting.