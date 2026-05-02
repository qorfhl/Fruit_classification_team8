# Fruit Image Classification — Team 8

> CNN-based 100-class fruit image classification project  
> Introduction to AI Programming | 2026 Spring Semester

---

## Team Members

| Name |
|------|
| Junyoung Kim |
| MinJun Kim |
| Daewook Jeon |

---

## Project Goal

Build and compare CNN-based image classifiers on the **Fruits-100** dataset (100 classes, 45,000 images).  
Three models are trained and compared:

1. **Baseline CNN** — lightweight custom CNN (from scratch)
2. **VGG16-style CNN** — deeper custom CNN (from scratch)
3. **ResNet50** — pretrained on ImageNet, fine-tuned on Fruits-100 ⭐ Best

---

## Dataset

- **Name**: Fruits-100
- **Source**: [Kaggle — marquis03/fruits-100](https://www.kaggle.com/datasets/marquis03/fruits-100)
- **Classes**: 100
- **Total images**: 45,000
- **Split**: 40,000 train / 5,000 validation (400 / 50 per class)
- **Imbalance ratio**: 1.00 (perfectly balanced)

### Download

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d marquis03/fruits-100
unzip fruits-100.zip -d imageset
```

Expected directory structure:

```
imageset/
├── train/
│   ├── apple/
│   ├── banana/
│   └── ...
└── val/
    ├── apple/
    ├── banana/
    └── ...
```

---

## 🛠 Environment & Dependencies

```
Python >= 3.8
torch >= 2.0
torchvision >= 0.15
scikit-learn
matplotlib
numpy
gradio (for demo only)
```

### Install

```bash
pip install torch torchvision scikit-learn matplotlib numpy gradio
```

---

## 🚀 Training

### Baseline CNN

```bash
python baseline.py \
  --train_dir ./imageset/train \
  --val_dir ./imageset/val \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-3 \
  --output_dir ./outputs/baseline
```

### VGG16-style CNN

```bash
python vgg16.py \
  --train_dir ./imageset/train \
  --val_dir ./imageset/val \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-3 \
  --output_dir ./outputs/vgg16
```

### ResNet50 (Pretrained) ⭐

```bash
python resnet50.py \
  --train_dir ./imageset/train \
  --val_dir ./imageset/val \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --save_model \
  --output_dir ./outputs/resnet50
```

---

## 📊 Evaluation

Each training script automatically generates the following in the output directory:

| File | Description |
|------|-------------|
| `learning_curve_*.png` | Loss / Accuracy / Macro-F1 curves |
| `confusion_matrix_*.png` | 100×100 confusion matrix |
| `classwise_f1_*.csv` | Per-class precision, recall, F1 |
| `correct_predictions.png` | Example correct predictions |
| `wrong_predictions.png` | Example wrong predictions |
| `class_distribution.png` | Training set class distribution |

---

## Inference Demo

Run the Gradio demo using the best ResNet50 checkpoint:

```bash
python demo.py
```

Then open the URL shown in the terminal (e.g., `http://127.0.0.1:7860`).

- **Input**: Upload any fruit image (JPG or PNG)
- **Output**: Top-3 predicted fruit classes with confidence scores

---

## 📈 Main Results

| Model | Val Accuracy | Val Macro-F1 | Params | Training Time |
|-------|-------------|--------------|--------|---------------|
| Baseline CNN | 41.42% | 0.4034 | ~45K | ~10.7 min |
| VGG16-style (scratch) | 48.74% | 0.4676 | ~134.7M | ~43.2 min |
| ResNet50 (pretrained) ⭐ | **82.02%** | **0.8199** | ~23.7M | ~24.1 min |

> Best checkpoint: ResNet50, Epoch 20

---

## Repository Structure

```
.
├── baseline.py          # Baseline CNN training script
├── vgg16.py             # VGG16-style CNN training script
├── resnet50.py          # ResNet50 training script
├── demo.py              # Gradio inference demo
├── README.md
└── outputs/
    ├── baseline/
    ├── vgg16/
    └── resnet50/
        └── best_model_resnet50.pth
```

---

## 📝 License

This project is for academic purposes only.
