# Emotions Detector

Real-time facial emotion recognition for the 01-edu / Tomorrow School computer-vision project.

The mandatory path trains a **CNN from scratch** on the supplied FER CSV dataset, evaluates it on a labeled test set, detects faces with OpenCV, converts them to **48x48 grayscale** crops, and predicts one of seven emotions from a webcam or recorded video.

## Emotions

| ID | Emotion |
|---:|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

## Project structure

```text
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── test_with_emotions.csv
├── results/
│   ├── model/
│   │   ├── final_emotion_model.keras
│   │   ├── final_emotion_model_arch.txt
│   │   ├── learning_curves.png
│   │   ├── tensorboard.png
│   │   ├── pre_trained_model.keras
│   │   └── pre_trained_model_architecture.txt
│   ├── preprocessing_test/
│   │   ├── input_video.mp4
│   │   ├── image0.png
│   │   └── ...
│   └── tensorboard/
├── scripts/
│   ├── common.py
│   ├── train.py
│   ├── validation_loss_accuracy.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── predict_live_stream.py
│   ├── report_metrics.py
│   ├── train_pretrained.py
│   └── adversarial_attack.py
├── requirements.txt
└── README.md
```

Generated artifacts do not exist until the corresponding command has been run.

## 1. Environment

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Linux / WSL:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For NVIDIA GPU training on Windows, use a TensorFlow-supported CUDA environment. WSL2 is usually the simplest route for recent TensorFlow releases. CPU training also works but is much slower.

## 2. Dataset

Download the project dataset supplied by the subject:

```text
https://assets.01-edu.org/ai-branch/project3/emotions-detector.zip
```

Put the CSV files in `data/`.

The mandatory training script reads only `data/train.csv`. The labeled test set is never used during training.

The loader expects the standard FER representation:

- `pixels`: 2304 space-separated grayscale pixel values
- `emotion` (or `label`): integer class in `0..6`

## 3. Mandatory model: CNN trained from scratch

Optional first iteration:

```bash
python ./scripts/train.py --profile baseline
```

Final model:

```bash
python ./scripts/train.py --profile final
```

Default final training configuration:

- stratified 85/15 train/validation split from `train.csv`
- random seed `42`
- 48x48 grayscale input
- CNN trained from random initialization
- training-only image augmentation
- Batch Normalization
- L2 regularization
- Dropout
- Adam optimizer
- `ModelCheckpoint`
- `EarlyStopping`
- `ReduceLROnPlateau`
- TensorBoard
- test set is not loaded

The final architecture contains four convolutional blocks with `64 / 128 / 256 / 256` filters followed by a dense classification head.

Generated mandatory files:

```text
results/model/final_emotion_model.keras
results/model/final_emotion_model_arch.txt
results/model/training_history.json
results/model/learning_curves.png
results/model/training_runs.jsonl
```

The project requirement is **more than 60% accuracy on the labeled test set**. The repository does not hardcode or claim an accuracy before real training is completed.

If minority classes need stronger weighting, compare a second run:

```bash
python ./scripts/train.py --profile final --class-weights
```

## 4. TensorBoard

Training automatically writes TensorBoard logs to `results/tensorboard/`.

```bash
tensorboard --logdir results/tensorboard
```

Inspect the real run and save the required screenshot as:

```text
results/model/tensorboard.png
```

The repository intentionally does not ship a fake placeholder screenshot.

## 5. Learning curves

Generated automatically by `train.py`, or regenerate them with:

```bash
python ./scripts/validation_loss_accuracy.py
```

Output:

```text
results/model/learning_curves.png
```

## 6. Test-set evaluation

Run exactly:

```bash
python ./scripts/predict.py
```

Expected output format:

```text
Accuracy on test set: 62.00%
```

The number above is only an example format. The command prints the actual saved model accuracy.

Optional inference benchmark:

```bash
python ./scripts/predict.py --benchmark
```

## 7. Face preprocessing audit

The face detector uses `cv2.CascadeClassifier` with OpenCV's bundled frontal-face cascade.

Webcam:

```bash
python ./scripts/preprocess.py --source 0 --preview
```

Recorded video:

```bash
python ./scripts/preprocess.py --source path/to/video.mp4 --preview
```

The default run targets at least 20 detected faces from a 20-second stream and saves:

```text
results/preprocessing_test/input_video.mp4
results/preprocessing_test/image0.png
results/preprocessing_test/image1.png
...
```

Every saved crop is detected by OpenCV, cropped around a face, converted to grayscale and resized to exactly `48x48`. The script fails instead of claiming success if it collects fewer than the requested number of faces.

## 8. Live emotion detection

Webcam:

```bash
python ./scripts/predict_live_stream.py --source 0
```

Recorded video fallback:

```bash
python ./scripts/predict_live_stream.py --source results/preprocessing_test/input_video.mp4
```

Press `q` to stop the preview. The terminal prints emotion/probability status at least once per second.

Live extras:

- multiple faces
- emotion + probability labels
- top-3 probability bars
- FPS display
- temporal EMA smoothing between overlapping face boxes

Disable smoothing:

```bash
python ./scripts/predict_live_stream.py --no-smoothing
```

Headless mode:

```bash
python ./scripts/predict_live_stream.py --headless
```

## 9. Extra evaluation

```bash
python ./scripts/report_metrics.py
```

Generated files:

```text
results/model/classification_report.txt
results/model/confusion_matrix.png
results/model/confident_mistakes.png
```

## 10. Bonus: pre-trained CNN

The mandatory `final_emotion_model.keras` remains a from-scratch CNN. The optional transfer-learning path is separate:

```bash
python ./scripts/train_pretrained.py
```

It uses MobileNetV2 with ImageNet weights, trains a new classifier head, then optionally fine-tunes the last 30 backbone layers.

Generated files:

```text
results/model/pre_trained_model.keras
results/model/pre_trained_model_architecture.txt
results/model/pre_trained_learning_curves.png
```

## 11. Bonus: hack the CNN (Happy -> Sad)

After the mandatory model has been trained:

```bash
python ./scripts/adversarial_attack.py
```

The script finds a real Happy test image predicted Happy with at least 90% confidence, keeps the CNN weights unchanged, and runs a targeted projected-gradient attack toward Sad while constraining the pixel delta.

Generated files:

```text
results/model/adversarial/original.png
results/model/adversarial/adversarial.png
results/model/adversarial/perturbation_amplified.png
results/model/adversarial/comparison.png
results/model/adversarial/metrics.txt
```

If the default perturbation is too weak, increase it explicitly and prefer the smallest successful value.

## 12. Reproducibility

Training fixes Python, NumPy and TensorFlow seeds and uses a deterministic stratified split with seed `42`. Every training run is appended to `results/model/training_runs.jsonl`.

## Notes inherited from vision-track

Only useful pieces were carried over:

- robust Windows webcam backend preference (`MSMF -> DSHOW -> ANY`)
- safe video-source opening
- OpenCV overlay/label rendering style
- FPS measurement approach

The multi-stream queues, scheduler, ByteTrack tracking, ROI/counting and Streamlit stack were intentionally not ported because they are unnecessary here.
