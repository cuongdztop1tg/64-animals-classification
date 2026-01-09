# 64-Animals Classification

The **64-Animals Classification** project is a Deep Learning application built with **PyTorch** for image classification, aiming to recognize **64 different animal categories** from input images.

The project is structured in a **modular, scalable, and maintainable** manner, making it suitable for academic study, experimentation, and extension with additional models or datasets.

---

## 1. Project Objectives

* Build a complete **image classification pipeline**
* Implement and experiment with ResNet architecture
* Train, evaluate, and compare model performance
* Visualize evaluation results (e.g., confusion matrix)
* Maintain clean, reusable, and well-organized code

---

## 2. Technologies Used

* **Python 3.x**
* **PyTorch** – deep learning framework
* **Torchvision** – image processing utilities and pretrained models
* **NumPy / Pandas** – data handling
* **Matplotlib / Seaborn** – visualization

---

## 3. Project Structure

```text
64-ANIMALS-CLASSIFICATION/
├── data/                       # Dataset storage (train / val / test)
├── models/                     # Saved trained models and checkpoints
├── src/
│   ├── core/                   # Core training and evaluation logic
│   ├── models/                 # Model architectures
│   │   └── ResNet/              # Custom ResNet implementation
│   │       ├── __init__.py
│   │       ├── residual_block.py
│   │       └── resnet.py
│   ├── scripts/                # Executable scripts
│   │   ├── prepare_data.py      # Dataset preprocessing
│   │   ├── train.py             # Model training
│   │   └── evaluate.py          # Model evaluation
│   ├── utils/                  # Utility functions
│   │   ├── save_model.py        # Model saving/loading helpers
│   │   ├── set_random_seed.py   # Reproducibility utilities
│   │   └── __init__.py
│   ├── app.py                  # Main application / inference entry
│   └── __init__.py
├── tests/                      # Unit and integration tests
├── venv/                       # Python virtual environment
├── .env                        # Environment variables
├── .gitignore
├── class_names.json            # Mapping of class indices to labels
├── confusion_matrix.png        # Evaluation visualization
├── README.md
└── requirements.txt            # Project dependencies
```

---

## 4. Dataset

* The dataset consists of images belonging to **64 animal classes**
* Images are expected to be organized in a standard folder-based format:

```text
data/
├── image/
│   ├── cat/
│   ├── dog/
│   └── ...
├── processed/
    └── train/
    └── test/
```

> Note: The dataset itself is not included in this repository.

---

## 5. Installation

### 5.1 Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 5.2 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 6. Usage

### 6.1 Prepare data

```bash
python src/scripts/prepare_data.py
```

### 6.2 Train a model

```bash
python src/scripts/train.py
```

You can configure:

* Model architecture
* Learning rate
* Batch size
* Number of epochs

inside the `src/core/config.py` file

### 6.3 Evaluate a trained model

```bash
python src/scripts/evaluate.py
```

Evaluation results may include:

* Accuracy
* Loss
* Confusion matrix

---
## 7. Reproducibility

To ensure reproducible results, the project provides a utility to fix random seeds:

```python
from utils.set_random_seed import set_seed
set_seed(42)
```

---

## 8. Results

* Trained models are saved in the `models/` directory
* Evaluation artifacts such as `confusion_matrix.png` are generated after evaluation

---

## 9. Future Improvements

* Add configuration management (YAML / argparse)
* Integrate experiment tracking (TensorBoard, Weights & Biases)
* Support additional architectures (AlexNet, EfficientNet)
* Deploy inference via REST API or web interface

---

## 10. License

This project is intended for **educational and research purposes**.

---

## 11. Author

Developed as part of a Deep Learning / Computer Vision learning project.
