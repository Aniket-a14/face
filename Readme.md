# Face Gender & Identity Recognition

This project provides a pipeline for training and running inference on a face dataset (e.g., UTKFace) to predict both gender and identity using deep learning with PyTorch and albumentations.

---

## Features

- **Dual-head model:** Predicts both gender (binary) and identity (multi-class) from face images.
- **Data augmentation:** Uses albumentations for robust training.
- **Training & inference scripts:** Easy-to-use scripts for training, single-image prediction, and real-time webcam prediction.
- **Identity mapping:** Handles mapping between dataset identity labels and model class indices.

---

## Project Structure

```
face/
│
├── src/
│   ├── augmentations.py      # Data augmentation and transforms
│   ├── dataset.py            # Custom PyTorch dataset
│   ├── model.py              # DualHeadFaceNet model definition
│   ├── train.py              # Training loop
│   ├── predict.py            # Single-image prediction
│   ├── realtime_predict.py   # Real-time webcam prediction
│   └── __init__.py
│
├── run_train.py              # Training entry point
├── parse_utkface.py          # Script to parse UTKFace dataset into CSV
├── utkface_processed.csv     # Processed dataset CSV (generated)
├── id_to_idx.json            # Identity-to-index mapping (generated)
├── model.pt                  # Trained model weights (generated)
├── .gitignore
└── README.md
```

---

## Setup

1. **Clone the repository and install dependencies:**
    ```sh
    git clone <repo-url>
    cd face
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```

2. **Prepare the dataset:**
    - Download the [UTKFace dataset](https://susanqq.github.io/UTKFace/).
    - Place images in `data/raw/UTKFace/`.

3. **Parse the dataset:**
    ```sh
    python parse_utkface.py
    ```
    This will generate `utkface_processed.csv`.

---

## Training

Run the training script:

```sh
python run_train.py
```

- This will train the model and save weights as `model.pt`.
- It will also save the identity-to-index mapping as `id_to_idx.json`.

---

## Inference

### Predict on a Single Image

```sh
python -m src.predict path/to/image.jpg
```
- Prints predicted gender and identity.

### Real-time Webcam Prediction

```sh
python -m src.realtime_predict
```
- Opens webcam and displays real-time gender prediction on detected faces.
- Press `q` to quit.

---

## Customization

- **Augmentations:** Edit `src/augmentations.py` to change training or validation transforms.
- **Model:** Edit `src/model.py` to modify the architecture.
- **Dataset:** Use `src/dataset.py` for custom dataset logic.

---

## Notes

- If you get `ModuleNotFoundError: No module named 'src'`, run scripts using `python -m src.scriptname` from the project root.
- If your webcam is not detected, try changing the camera index in `src/realtime_predict.py` (`cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`).

---

## Requirements

- Python 3.8+
- torch
- torchvision
- albumentations
- opencv-python
- pandas
- tqdm
- pillow

---

##