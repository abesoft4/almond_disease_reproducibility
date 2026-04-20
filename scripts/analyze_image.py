from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from load_inference_artifacts import (
    PACKAGE_DIR,
    get_explainability_config,
    load_base_model,
    load_model_manifest,
    predict_calibrated,
)


IMAGE_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run calibrated prediction and explainability analysis on a single image."
    )
    parser.add_argument("image", type=Path, help="Path to an input image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_DIR / "outputs" / "single_image",
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=64,
        help="Number of interpolation steps for Integrated Gradients.",
    )
    return parser.parse_args()


def load_image(image_path: Path, target_size: tuple[int, int] = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB").resize(target_size)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    batch = np.expand_dims(image_np, axis=0)
    return image_np, batch


def find_last_conv2d(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM.")


def make_gradcam_heatmap(image_batch: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str, class_index: int) -> np.ndarray:
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads[tf.newaxis, tf.newaxis, :], axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) > 0:
        heatmap = heatmap / max_value
    return heatmap.numpy()


def overlay_heatmap(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], image_np.shape[:2]).numpy()[..., 0]
    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(np.clip(heatmap_resized, 0, 1))[..., :3]
    return np.clip((1 - alpha) * image_np + alpha * heatmap_rgb, 0, 1)


def integrated_gradients(model: tf.keras.Model, image_batch: np.ndarray, class_index: int, steps: int = 64) -> np.ndarray:
    baseline = tf.zeros_like(image_batch)
    image = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    grads_accum = tf.zeros_like(image)

    for alpha in alphas:
        interpolated = baseline + alpha * (image - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            predictions = model(interpolated, training=False)
            target = predictions[:, class_index]
        grads_accum += tape.gradient(target, interpolated)

    avg_grads = grads_accum / tf.cast(steps + 1, tf.float32)
    attributions = (image - baseline) * avg_grads
    return tf.reduce_mean(tf.abs(attributions), axis=-1)[0].numpy()


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values -= values.min()
    denom = values.max()
    if denom > 0:
        values /= denom
    return values


def save_panel(image_np: np.ndarray, gradcam_overlay: np.ndarray, ig_map: np.ndarray, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(image_np)
    axes[0].set_title("Input")
    axes[1].imshow(gradcam_overlay)
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(image_np)
    axes[2].imshow(ig_map, cmap="inferno", alpha=0.55)
    axes[2].set_title("Integrated Gradients")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_image(image_path: Path, output_dir: Path, ig_steps: int = 64, output_stem: str | None = None) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_model_manifest()
    explainability = get_explainability_config()
    class_names = explainability["class_names"]
    base_model = load_base_model(compile=False)
    last_conv_layer = explainability["last_conv_layer"] or find_last_conv2d(base_model)

    image_np, image_batch = load_image(image_path)
    raw_probs = base_model.predict(image_batch, verbose=0)[0]
    cal_probs = predict_calibrated(image_batch, model=base_model, verbose=0)[0]

    pred_idx = int(np.argmax(raw_probs))
    pred_label = class_names[pred_idx]
    gradcam = make_gradcam_heatmap(image_batch, base_model, last_conv_layer, pred_idx)
    ig_map = normalize_map(integrated_gradients(base_model, image_batch, pred_idx, steps=ig_steps))
    gradcam_overlay = overlay_heatmap(image_np, gradcam)

    stem = output_stem or image_path.stem
    panel_path = output_dir / f"{stem}_explainability.png"
    json_path = output_dir / f"{stem}_prediction.json"
    save_panel(image_np, gradcam_overlay, ig_map, panel_path, title=f"Prediction: {pred_label}")

    result = {
        "image": str(image_path.resolve()),
        "checkpoint": explainability["checkpoint"],
        "last_conv_layer": last_conv_layer,
        "temperature": manifest["temperature_scaling"]["temperature"],
        "predicted_class_index": pred_idx,
        "predicted_class_label": pred_label,
        "raw_probabilities": {name: float(raw_probs[i]) for i, name in enumerate(class_names)},
        "temperature_scaled_probabilities": {name: float(cal_probs[i]) for i, name in enumerate(class_names)},
        "artifacts": {
            "prediction_json": str(json_path),
            "explainability_panel": str(panel_path),
        },
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    return result


def main() -> None:
    args = parse_args()
    result = analyze_image(args.image, args.output_dir, ig_steps=args.ig_steps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()