from __future__ import annotations

import json
from pathlib import Path
import tempfile
import zipfile


PACKAGE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PACKAGE_DIR / "model"
METADATA_DIR = PACKAGE_DIR / "metadata"
MANIFEST_PATH = METADATA_DIR / "model_manifest.json"
PREPROCESSING_PATH = METADATA_DIR / "preprocessing.json"


def load_model_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_preprocessing_manifest() -> dict:
    with PREPROCESSING_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_class_names() -> list[str]:
    manifest = load_model_manifest()
    classes_path = PACKAGE_DIR / manifest["artifacts"]["classes"]
    with classes_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _aug_pipeline(level: str = "mild"):
    import tensorflow as tf

    keras = tf.keras
    L = keras.layers
    if level == "mild":
        return keras.Sequential(
            [
                L.RandomFlip("horizontal"),
                L.RandomRotation(0.05),
                L.RandomZoom(0.1),
                L.RandomContrast(0.1, value_range=(0.0, 1.0)),
            ],
            name="aug_mild",
        )
    if level == "strong":
        return keras.Sequential(
            [
                L.RandomFlip("horizontal_and_vertical"),
                L.RandomRotation(0.15),
                L.RandomZoom(0.2),
                L.RandomContrast(0.2, value_range=(0.0, 1.0)),
                L.RandomBrightness(0.15, value_range=(0.0, 1.0)),
                L.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
            ],
            name="aug_strong",
        )
    return keras.Sequential([], name="aug_none")


def _preprocess_for_backbone(x, backbone: str):
    import tensorflow as tf

    L = tf.keras.layers
    if backbone == "mobilenetv2":
        return L.Lambda(lambda t: t * 2.0 - 1.0, name="pre_mnv2")(x)
    if backbone == "resnet50":
        import tensorflow.keras.applications.resnet50 as resnet50

        return L.Lambda(lambda t: resnet50.preprocess_input(t * 255.0), name="pre_resnet")(x)
    if backbone == "efficientnetv2b0":
        import tensorflow.keras.applications.efficientnet_v2 as efficientnet_v2

        return L.Lambda(lambda t: efficientnet_v2.preprocess_input(t * 255.0), name="pre_effv2")(x)
    return x


def _rebuild_best_model():
    import tensorflow as tf

    manifest = load_model_manifest()
    cfg = manifest["best_configuration"]
    class_names = load_class_names()
    keras = tf.keras
    L = keras.layers
    image_size = (224, 224)

    backbone = cfg["backbone"].lower().replace("-", "").replace("_", "")

    inputs = L.Input(shape=image_size + (3,))
    x = _aug_pipeline(cfg["augmentation"])(inputs)
    x = _preprocess_for_backbone(x, backbone)

    if backbone == "efficientnetv2b0":
        base = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights=None,
            input_tensor=x,
            pooling="avg",
        )
    elif backbone == "mobilenetv2":
        base = keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_tensor=x,
            pooling="avg",
        )
    elif backbone == "resnet50":
        base = keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_tensor=x,
            pooling="avg",
        )
    else:
        raise ValueError(f"Unsupported backbone in manifest: {cfg['backbone']}")

    x = L.Dropout(cfg["dropout"])(base.output)
    outputs = L.Dense(len(class_names), activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_{cfg['augmentation']}_do{cfg['dropout']}")


def _extract_keras_weights(keras_zip_path: Path) -> Path:
    with zipfile.ZipFile(keras_zip_path, "r") as archive:
        candidates = [name for name in archive.namelist() if name.endswith(".weights.h5")]
        if not candidates:
            raise FileNotFoundError(f"No *.weights.h5 found inside {keras_zip_path}")
        inner_path = candidates[0]
        target_path = Path(tempfile.gettempdir()) / (keras_zip_path.stem + ".weights.h5")
        with archive.open(inner_path) as src, open(target_path, "wb") as dst:
            dst.write(src.read())
    return target_path


def _load_weights_only_model(compile: bool = False):
    model = _rebuild_best_model()
    manifest = load_model_manifest()
    checkpoint_path = PACKAGE_DIR / manifest["artifacts"]["checkpoint"]
    weights_path = _extract_keras_weights(checkpoint_path)
    try:
        model.load_weights(weights_path)
    except Exception:
        model.load_weights(weights_path, by_name=True, skip_mismatch=False)
    return model


def load_base_model(compile: bool = False):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required to load the model checkpoint.") from exc

    manifest = load_model_manifest()
    checkpoint_path = PACKAGE_DIR / manifest["artifacts"]["checkpoint"]
    try:
        return tf.keras.models.load_model(checkpoint_path, compile=compile, safe_mode=False)
    except TypeError:
        try:
            return tf.keras.models.load_model(checkpoint_path, compile=compile)
        except Exception:
            return _load_weights_only_model(compile=compile)
    except Exception:
        return _load_weights_only_model(compile=compile)


def temperature_scale_probabilities(probabilities, temperature: float, eps: float = 1e-8):
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("NumPy is required to apply temperature scaling.") from exc

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"Expected a 2D probability array, got shape {probs.shape}")
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    clipped = np.clip(probs, eps, 1.0)
    logits = np.log(clipped) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def predict_calibrated(inputs, model=None, **predict_kwargs):
    manifest = load_model_manifest()
    base_model = model if model is not None else load_base_model(compile=False)
    probabilities = base_model.predict(inputs, **predict_kwargs)
    return temperature_scale_probabilities(probabilities, manifest["temperature_scaling"]["temperature"])


def get_explainability_config() -> dict:
    manifest = load_model_manifest()
    return {
        "checkpoint": str((PACKAGE_DIR / manifest["artifacts"]["checkpoint"]).resolve()),
        "last_conv_layer": manifest["best_configuration"]["last_conv_layer"],
        "class_names": load_class_names(),
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "manifest": load_model_manifest(),
                "preprocessing": load_preprocessing_manifest(),
            },
            indent=2,
        )
    )