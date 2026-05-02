# Foundation Models for Information Extraction

A hands-on notebook covering key concepts in foundation models for information extraction — including ranking evaluation with Precision@k and multiclass text classification using BERT and its variants.

---

## Table of Contents

1. [Overview](#overview)
2. [Topics Covered](#topics-covered)
   - [Precision@k](#1-precisionk)
   - [BERT Multiclass Text Classification](#2-bert-multiclass-text-classification)
3. [Dataset](#dataset)
4. [Model Configuration](#model-configuration)
5. [Training Results](#training-results)
6. [Test Set Performance](#test-set-performance)
7. [Sample Predictions](#sample-predictions)
8. [Getting Started](#getting-started)
9. [Usage](#usage)
10. [Results Summary](#results-summary)
11. [Author](#author)
12. [References](#references)

---

## Overview

This notebook explores foundation models applied to information extraction tasks. It covers two core topics: manual computation of the **Precision@k** evaluation metric for ranked retrieval, and fine-tuning **BERT** (`bert-base-uncased`) for **multiclass text classification** on a research paper conference dataset. The notebook includes training, evaluation, model saving, and inference on custom paper titles.

---

## Topics Covered

### 1. Precision@k

Computes **Precision@3** for a 10-sample dataset where each sample has predicted scores and binary ground truth labels.

**Formula:**

$$\text{Precision@k} = \frac{\text{True Positives in top-}k\text{ predictions}}{k}$$

**Dataset used:**

| Sample | Predicted Scores | Ground Truth Labels |
|--------|-----------------|---------------------|
| 1 | [0.9, 0.1, 0.3, 0.6, 0.2] | [1, 0, 0, 1, 0] |
| 2 | [0.2, 0.8, 0.7, 0.1, 0.4] | [0, 1, 1, 0, 0] |
| 3 | [0.5, 0.2, 0.9, 0.4, 0.6] | [0, 0, 1, 1, 0] |
| 4 | [0.1, 0.7, 0.2, 0.5, 0.8] | [0, 1, 0, 0, 1] |
| 5 | [0.6, 0.9, 0.4, 0.3, 0.2] | [1, 0, 0, 0, 0] |
| 6 | [0.3, 0.4, 0.8, 0.6, 0.1] | [0, 0, 1, 1, 0] |
| 7 | [0.7, 0.2, 0.1, 0.9, 0.4] | [0, 1, 0, 1, 0] |
| 8 | [0.4, 0.5, 0.3, 0.7, 0.2] | [1, 0, 0, 1, 0] |
| 9 | [0.2, 0.6, 0.8, 0.3, 0.9] | [0, 1, 1, 0, 1] |
| 10 | [0.5, 0.3, 0.4, 0.7, 0.2] | [0, 0, 0, 1, 0] |

**Results (k=3):**

| Sample | Top 3 Positions | Ground Truth | TP | Precision@3 |
|--------|-----------------|--------------|----|-------------|
| 1 | [0, 3, 2] | [1, 1, 0] | 2 | 0.6667 |
| 2 | [1, 2, 4] | [1, 1, 0] | 2 | 0.6667 |
| 3 | [2, 4, 0] | [1, 0, 0] | 1 | 0.3333 |
| 4 | [4, 1, 3] | [1, 1, 0] | 2 | 0.6667 |
| 5 | [1, 0, 2] | [0, 1, 0] | 1 | 0.3333 |
| 6 | [2, 3, 1] | [1, 1, 0] | 2 | 0.6667 |
| 7 | [3, 0, 4] | [1, 0, 0] | 1 | 0.3333 |
| 8 | [3, 1, 0] | [1, 0, 1] | 2 | 0.6667 |
| 9 | [4, 2, 1] | [1, 1, 1] | 3 | 1.0000 |
| 10 | [3, 0, 2] | [1, 0, 0] | 1 | 0.3333 |

**Mean Precision@3: `0.5667`**

---

### 2. BERT Multiclass Text Classification

Fine-tunes `bert-base-uncased` to classify research paper titles into **5 conference categories**.

#### Tasks Implemented

| Task | Description |
|------|-------------|
| **a** | Fine-tune BERT for 5-class classification (80/20 train-test split) |
| **b** | Train on 100 samples from the dataset |
| **c** | Save the trained model to `bert_conference_model.pt` |
| **d** | Plot training loss and accuracy per epoch |
| **e** | Test on custom paper titles |
| **f** | Evaluate on the test set with full classification report |
| **g** | Compare with alternative models (`distilbert-base-uncased`, `roberta-base`) |
| **h** | Experiment with larger training data (up to full dataset) |

---

## Dataset

**Conference Dataset** — 2,507 research paper titles manually classified into 5 conference categories.

- Source: [title_conference.csv](https://raw.githubusercontent.com/susanli2016/NLP-with-Python/master/data/title_conference.csv)
- Columns: `Title` (text), `Conference` (label, 5 classes: 0–4)
- Split: **80% training / 20% testing**

---

## Model Configuration

| Hyperparameter | Value |
|---------------|-------|
| Model | `bert-base-uncased` |
| Learning Rate | `2e-5` |
| Batch Size | `16` |
| Max Sequence Length | `256` |
| Epochs | `10` |
| Optimizer | AdamW with linear warmup |
| Random Seed | `42` |

---

## Training Results

Training was performed on a GPU (NVIDIA T4) via Google Colab.

| Epoch | Training Loss | Training Accuracy |
|-------|--------------|-------------------|
| 1 | 1.0398 | 0.6045 |
| 2 | 0.4842 | 0.8404 |
| 3 | 0.2929 | 0.9112 |
| 4 | 0.1642 | 0.9531 |
| 5 | 0.0897 | 0.9751 |
| 6 | 0.0601 | 0.9825 |
| 7 | 0.0297 | 0.9925 |
| 8 | 0.0142 | 0.9980 |
| 9 | 0.0078 | 0.9990 |
| 10 | 0.0077 | 0.9990 |

Training loss and accuracy curves are plotted inline during execution.

---

## Test Set Performance

Evaluated on 502 test samples (20% of dataset):

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.79 | 0.81 | 0.80 | 108 |
| 1 | 0.90 | 0.89 | 0.89 | 178 |
| 2 | 0.82 | 0.73 | 0.77 | 67 |
| 3 | 0.65 | 0.73 | 0.68 | 73 |
| 4 | 0.68 | 0.66 | 0.67 | 76 |
| **Accuracy** | | | **0.79** | **502** |
| Macro Avg | 0.77 | 0.76 | 0.76 | 502 |
| Weighted Avg | 0.80 | 0.79 | 0.79 | 502 |

**Overall Accuracy: `79.28%`**

---

## Sample Predictions

| Paper Title | Predicted Class |
|-------------|----------------|
| Deep Learning for Natural Language Understanding | 4 |
| Blockchain Applications in Healthcare | 0 |
| Cybersecurity in the Age of IoT | 0 |

---

## Getting Started

### Prerequisites

```bash
pip install torch transformers pandas scikit-learn matplotlib openpyxl tqdm
```

### Clone and Run

```bash
git clone <your-repo-url>
cd <repo-folder>
jupyter notebook FoundationModelsForInformationExtraction.ipynb
```

### Download the Dataset

```bash
wget https://raw.githubusercontent.com/susanli2016/NLP-with-Python/master/data/title_conference.csv
```

> Update the `csv_file` path in the `main()` function to point to your local copy.

---

## Usage

1. Open `FoundationModelsForInformationExtraction.ipynb` in Jupyter or Google Colab.
2. Run **Section 1** cells for the Precision@k manual calculation.
3. Run **Section 2** cells sequentially for BERT fine-tuning:
   - Install dependencies
   - Import libraries
   - Execute the `main()` function to train, evaluate, and save the model
4. The trained model is saved as `bert_conference_model.pt`.
5. Use `predict_text(model, tokenizer, "your title here")` to classify a new paper title.

### Predict a Custom Title

```python
pred_class = predict_text(model, tokenizer, "Federated Learning for Privacy-Preserving AI")
print(f"Predicted Conference Class: {pred_class}")
```

---

## Results Summary

| Task | Key Result |
|------|-----------|
| Precision@3 (mean) | **0.5667** across 10 samples |
| BERT Training Accuracy | **99.90%** at epoch 10 |
| BERT Test Accuracy | **79.28%** on 502 samples |
| Best class F1 | **0.89** (Class 1) |
| Model saved as | `bert_conference_model.pt` |

---

## Author

**Sayed Hamidullah Fazlly**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/sayed-hamidullah-fazlly-382489170)

---

## References

- Devlin, J. et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). NAACL.
- HuggingFace `transformers` documentation: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- Conference Dataset: [susanli2016/NLP-with-Python](https://github.com/susanli2016/NLP-with-Python/blob/master/data/title_conference.csv)
- Manning, C. et al. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
