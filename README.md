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
| Saved checkpoint | [`out/emotion_model.h5`](out/emotion_model.h5) is retained in the repository. It is excluded from Pages and was not executed or converted in this completion pass. |

To regenerate the history: `python3 scripts/export-history.py`; to check it without overwriting: `python3 scripts/export-history.py --check`. The new dependency-free log reader fixes brittle column parsing and reversed epoch ordering in the plotting path. Tests check all historical layer shapes and parameters, known numeric image operations, log chronology and malformed log rejection. The Pages workflow tests changes and publishes only `demo/`.

## Historical training path and limitations

The original environment suggested Python 3.7 with the pinned TensorFlow 2.4-era stack in [`requirements.txt`](requirements.txt). Those historical packages are preserved for provenance; this is not a tested or recommended modern environment. The FER2013 CSV is absent and must be obtained under its own terms from the linked [dataset source](https://www.kaggle.com/ashishpatel26/facial-expression-recognitionferchallenge). It is not redistributed here.

The original workflow runs `python3 main.py` from `src/`, expects `data/fer2013.csv`, and writes model/results under `out/`. It defaults to training the selected Happy/Neutral labels with a two-output softmax and the source's binary-crossentropy loss. The code uses **PrivateTest for validation and PublicTest for final evaluation**; the old log labels that final evaluation “Validation.” These naming and methodological choices must be understood before comparing the metrics. No accuracy generalization, calibration, fairness, or internal-emotion inference is established.

Training was not rerun in this completion pass. Modern dependency migration, independent dataset evaluation and checkpoint execution remain unverified. The legacy webcam demo is now off by default; it requires deliberate local configuration plus a compatible model/runtime and is separate from the browser example. Public coursework reports remain in `documents/`; their historical claims are not new results.
