# Emotion Detection CNN

A historical 2021 TensorFlow/Keras learning project, now accompanied by an interactive explanation of its architecture, preprocessing and recorded training run.

**[Open the CNN Explorer](https://elliottbarnes.github.io/emotion-detection-cnn/)**

Change a synthetic pattern, inspect grayscale normalization and a hand-selected convolution kernel, follow tensor shapes through the actual model architecture, and scrub through 49 recorded training epochs. The browser does not load the checkpoint, infer emotions, collect images, access a camera, or claim a new evaluation.

## Run the complete browser example

Requires Node.js 24+ and Python 3. No packages, models, or datasets are required for this path.

```sh
node --test
node scripts/check-demo.mjs
python3 -m http.server 4177 --bind 127.0.0.1 --directory demo
```

Open **http://localhost:4177**. Browser modules require HTTP rather than `file://`.

## What is real, and what is illustrative

| Explorer surface | Source and scope |
| --- | --- |
| Architecture | Layer order, shape arithmetic, and parameter counts follow [`src/model.py`](src/model.py), including redundant Flatten layers. At 48 × 48, all 26 layer outputs match the model summary in the historical log: 156,898 total parameters. Other input sizes are hypothetical architecture calculations. |
| Training history | Numeric epoch records are extracted from [`out/out.txt`](out/out.txt), a run dated 11 March 2021. [`demo/history.json`](demo/history.json) includes the source SHA-256. Values are historical observations, not newly reproduced metrics. |
| Preprocessing | Synthetic grayscale values are divided by 255 as in [`src/data.py`](src/data.py). Three hand-selected kernels demonstrate valid convolution, ReLU and max pooling; they are not learned filters. |
| Saved checkpoint | [`out/emotion_model.h5`](out/emotion_model.h5) is retained unchanged and excluded from Pages. Its weights and synthetic-input predictions are checked on the modern Python runtime; this is not an independent accuracy evaluation. |

To regenerate the history: `python3 scripts/export-history.py`; to check it without overwriting: `python3 scripts/export-history.py --check`. The new dependency-free log reader fixes brittle column parsing and reversed epoch ordering in the plotting path. Tests check all historical layer shapes and parameters, known numeric image operations, log chronology and malformed log rejection. The Pages workflow tests changes and publishes only `demo/`.

## Historical training path and limitations

The original environment used Python 3.7 and TensorFlow 2.4-era packages. Those old pins remain in Git history. [`requirements.txt`](requirements.txt) now pins a patched environment for **Python 3.12**, including TensorFlow 2.21.0 and Keras 3.15.1. Use a fresh virtual environment so obsolete packages from the original installation do not remain installed:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s test_ml -p '*_test.py' -v
```

On Windows, activate with `.venv\Scripts\activate`. The ML checks use synthetic CSV data, exercise normalization, augmentation, cached arrays, one training epoch with callbacks, checkpoint saving/reloading, and predictions from the unchanged historical checkpoint. They do not use a camera or download a dataset. The browser/log checks above still need no ML dependencies.

The FER2013 CSV is absent and must be obtained under its own terms from the linked [dataset source](https://www.kaggle.com/ashishpatel26/facial-expression-recognitionferchallenge). It is not redistributed here.

The original workflow runs `python3 main.py` from `src/`, expects `data/fer2013.csv`, and writes model/results under `out/`. It defaults to training the selected Happy/Neutral labels with a two-output softmax and the source's binary-crossentropy loss. The code uses **PrivateTest for validation and PublicTest for final evaluation**; the old log labels that final evaluation “Validation.” These naming and methodological choices must be understood before comparing the metrics. No accuracy generalization, calibration, fairness, or internal-emotion inference is established.

The modern runtime was checked on Python 3.12.14/macOS ARM64 with synthetic data; full FER2013 training, historical accuracy, and live webcam behavior have not been revalidated. Load only trusted local checkpoints: `compile=False` skips obsolete optimizer state for inference and does not make arbitrary HDF5 models safe. The saved checkpoint records categorical cross-entropy, while the current source uses binary cross-entropy; that historical difference is preserved. The legacy webcam demo remains off by default and requires deliberate local configuration. Public coursework reports remain in `documents/`; their historical claims are not new results.
