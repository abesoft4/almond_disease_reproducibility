from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analyze_image import analyze_image
from load_inference_artifacts import PACKAGE_DIR


DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run calibrated prediction and explainability analysis for every image in a folder."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing images to analyze.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_DIR / "outputs" / "batch",
        help="Directory where per-image outputs and summaries will be written.",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=64,
        help="Number of interpolation steps for Integrated Gradients.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on the number of images to process.",
    )
    return parser.parse_args()


def iter_images(input_dir: Path):
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTENSIONS:
            yield path


def make_output_stem(image_path: Path, root_dir: Path) -> str:
    rel = image_path.resolve().relative_to(root_dir.resolve())
    parts = list(rel.with_suffix("").parts)
    return "__".join(parts)


def write_summary_csv(results: list[dict], output_path: Path) -> None:
    fieldnames = [
        "image",
        "predicted_class_label",
        "predicted_class_index",
        "temperature",
        "top_raw_class",
        "top_raw_probability",
        "top_calibrated_class",
        "top_calibrated_probability",
        "prediction_json",
        "explainability_panel",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            raw_probs = result["raw_probabilities"]
            cal_probs = result["temperature_scaled_probabilities"]
            top_raw_class = max(raw_probs, key=raw_probs.get)
            top_cal_class = max(cal_probs, key=cal_probs.get)
            writer.writerow(
                {
                    "image": result["image"],
                    "predicted_class_label": result["predicted_class_label"],
                    "predicted_class_index": result["predicted_class_index"],
                    "temperature": result["temperature"],
                    "top_raw_class": top_raw_class,
                    "top_raw_probability": raw_probs[top_raw_class],
                    "top_calibrated_class": top_cal_class,
                    "top_calibrated_probability": cal_probs[top_cal_class],
                    "prediction_json": result["artifacts"]["prediction_json"],
                    "explainability_panel": result["artifacts"]["explainability_panel"],
                }
            )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(input_dir))
    if args.max_images is not None:
        images = images[: args.max_images]
    if not images:
        raise FileNotFoundError(f"No supported images found under {input_dir}")

    results = []
    for image_path in images:
        output_stem = make_output_stem(image_path, input_dir)
        result = analyze_image(image_path, output_dir, ig_steps=args.ig_steps, output_stem=output_stem)
        results.append(result)
        print(f"Processed {image_path}")

    summary_json = output_dir / "batch_summary.json"
    summary_csv = output_dir / "batch_summary.csv"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    write_summary_csv(results, summary_csv)

    print(
        json.dumps(
            {
                "processed_images": len(results),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "batch_summary_json": str(summary_json),
                "batch_summary_csv": str(summary_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()