# Face Gender Recognition

This project provides a pipeline for training and running inference on a face dataset (with images organized in `data/raw/male/` and `data/raw/female/`) to predict gender using deep learning with PyTorch and albumentations.

---

## Features

- **Gender prediction** from face images using a ResNet18-based model.
- **Data augmentation** for robust training.
- **Training, single-image prediction, and real-time webcam prediction** scripts.
- **Automatic dataset parsing** from folder structure.

---

## Project Structure

```
face/
│
├── src/
│   ├── augmentations.py      # Data augmentation and transforms
│   ├── dataset.py            # Custom PyTorch dataset
│   ├── model.py              # DualHeadFaceNet model definition (gender head used)
│   ├── train.py              # Training loop
│   ├── predict.py            # Single-image gender prediction
│   ├── realtime_predict.py   # Real-time webcam gender prediction
│   └── __init__.py
│
├── run_train.py              # Training entry point
├── parse_utkface.py          # Script to parse dataset folders into CSV
├── face_processed.csv        # Processed dataset CSV (generated)
├── id_to_idx.json            # Identity-to-index mapping (generated, not used for gender-only)
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
    - Place your images in `data/raw/male/` and `data/raw/female/` folders.

3. **Parse the dataset:**
    ```sh
    python parse_utkface.py
    ```
    This will generate `face_processed.csv`.

---

## Training

Run the training script:

```sh
python run_train.py
```

- This will train the model and save weights as `model.pt`.

---

## Inference

### Predict on a Single Image

```sh
python -m src.predict path/to/image.jpg
```
- Prints predicted gender.

### Real-time Webcam Gender Prediction

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

## License

Apache License 2.0
---