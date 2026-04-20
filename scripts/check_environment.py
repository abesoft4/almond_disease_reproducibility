from __future__ import annotations

import json

from load_inference_artifacts import load_model_manifest, load_preprocessing_manifest


def main() -> None:
    import keras
    import numpy as np
    import tensorflow as tf

    devices = [device.name for device in tf.config.list_physical_devices()]
    report = {
        "python_runtime": {
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
        },
        "tensorflow_visible_devices": devices,
        "model_manifest": load_model_manifest(),
        "preprocessing_manifest": load_preprocessing_manifest(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()