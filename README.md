# Melanoma Detection with Deep Learning

A proof-of-concept skin lesion classifier that predicts whether a dermoscopic image shows a **benign** lesion or **melanoma**. It uses transfer learning on a pre-trained **ResNet-50** in **PyTorch**, with a **Streamlit** web app for uploading an image and getting a prediction.

Developed as part of the MSc Artificial Intelligence, University of Greater Manchester.

> ⚠️ **Educational project only.** This is not a medical device and must not be used to diagnose or rule out skin cancer. Anyone concerned about a skin lesion should see a doctor.

---

## Approach

- **Model:** ResNet-50 pre-trained on ImageNet, with a new classification head (2048 → 512 → ReLU → Dropout 0.5 → 2 classes)
- **Training:** the whole network is fine-tuned with Adam (learning rate 0.0001), batch size 16, for 5 epochs
- **Class weighting:** melanoma errors are weighted 2.5× in the loss, because missing a melanoma is more serious than a false alarm
- **Augmentation:** random horizontal flips and rotation (±20°) on the training images
- **Pre-processing:** images resized to 224 × 224 and normalised with ImageNet statistics, the same at training and prediction time

---

## Results

Evaluated on the held-out test set (`python evaluate.py`):

| | Precision | Recall | F1-score |
|---|---|---|---|
| Benign | 1.00 | 0.67 | 0.80 |
| Melanoma | 0.75 | **1.00** | 0.86 |
| **Accuracy** | | | **83% (5 of 6)** |

All melanoma images were correctly identified; one benign lesion was flagged as melanoma. That is the trade-off the class weighting was designed for.

**Limitations:** the dataset is very small (12 images per class for training, 3 for validation and 3 for testing), so these results demonstrate that the pipeline works rather than measuring real-world accuracy. A meaningful evaluation would need thousands of images, for example from the full ISIC Archive, plus cross-validation.

---

## Getting started

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/aniqa38/Cassifies-skin-lesion-images-as-benign-or-melanoma.git
cd Cassifies-skin-lesion-images-as-benign-or-melanoma
pip install -r requirements.txt
```

### 2. Get the trained model

Download `melanoma_model.pth` from this repository's **Releases** page and place it in the project folder. Alternatively, train your own (step 3).

### 3. Train (optional)

```bash
python train.py
```

### 4. Evaluate

```bash
python evaluate.py
```

### 5. Run the web app

```bash
streamlit run app.py
```

Upload a dermoscopic image to see the predicted class and confidence.

---

## Project structure

| File | Purpose |
|---|---|
| `model.py` | ResNet-50 model with the custom classification head |
| `train.py` | Fine-tunes the model and saves `melanoma_model.pth` |
| `evaluate.py` | Confusion matrix and classification report on the test set |
| `app.py` | Streamlit web app for predictions |
| `dataset/` | Sample images split into `train/`, `val/` and `test/`, each with `benign/` and `melanoma/` folders |

---

## Possible improvements

- Train on a much larger dataset, such as the ISIC 2019/2020 challenge data
- Add stronger augmentation (colour jitter, zoom) and early stopping based on validation loss
- Report ROC-AUC and sensitivity at a fixed specificity, which are standard for medical screening tasks
- Add Grad-CAM heatmaps to show which part of the image the model relied on

---

## Data credit

Images are from the [ISIC Archive](https://www.isic-archive.com) (International Skin Imaging Collaboration), released under the CC-0 licence.

## Author

**Aniqa Arooj**, MSc Artificial Intelligence, University of Greater Manchester
