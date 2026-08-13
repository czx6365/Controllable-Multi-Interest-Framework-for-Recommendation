# Multi-Interest Recommendation Models

**A TensorFlow/Keras study implementation of sequential and multi-interest recommendation models.**

This repository is a **reproduction / learning implementation**, not a new recommendation paper. It is based on ideas from *Controllable Multi-Interest Framework for Recommendation* (KDD 2020) and related multi-interest recommendation work.

The purpose of the repository is to study how a single user behavior sequence can be represented by multiple latent interests and then used for retrieval-oriented recommendation.

> **Scope note for readers:** this repository is separate from my later work on controllable generative recommendation. It should not be interpreted as the codebase for discrete-diffusion / generative recommendation research.

## What Is Implemented

The current code provides several sequential recommendation baselines and multi-interest models through a common TensorFlow/Keras interface:

| Model | Main idea |
| --- | --- |
| `DNN` | pooled history embedding followed by feed-forward layers |
| `GRU4REC` | GRU-based sequential user representation |
| `MIND` | capsule-network-based multi-interest extraction |
| `ComiRec-DR` | dynamic-routing multi-interest representation |
| `ComiRec-SA` | self-attention-based multi-interest representation |

The model factory in [`model.py`](model.py) exposes these implementations through a shared training interface.

## Why Multi-Interest Recommendation?

A single embedding can be too restrictive for users whose histories contain several unrelated interests.

For example:

```text
User history
├── machine learning books
├── running shoes
├── coffee equipment
└── travel guides
```

Compressing all of these behaviors into one vector can blur distinct intent. Multi-interest models instead learn several user vectors:

```text
behavior sequence
      ↓
sequence encoder
      ↓
multiple interest embeddings
      ↓
interest-aware retrieval
      ↓
Top-K recommendations
```

The repository is useful for comparing different mechanisms for extracting those latent interests.

## Core Components

### `model.py`

Implements the recommendation models and shared `BaseModel` interface.

Key pieces include:

- item embeddings;
- sequence masking;
- DNN and GRU encoders;
- capsule-network dynamic routing;
- multi-head interest extraction;
- hard interest readout;
- model save/load utilities.

### `train.py`

Contains the training and evaluation pipeline.

The evaluation path includes:

- item embedding export;
- FAISS inner-product retrieval when available;
- NumPy retrieval fallback;
- Recall;
- NDCG;
- Hit Rate;
- recommendation diversity based on item categories.

### `data_iterator.py`

Loads sequential recommendation data and constructs history / mask inputs for the models.

### `mostpop.py`

Provides a popularity-oriented baseline for comparison.

## Repository Structure

```text
.
├── README.md
├── data_iterator.py
├── model.py
├── mostpop.py
├── train.py
└── requirements
```

## Data

The original study data can be downloaded from the link referenced by the previous version of this repository:

```bash
wget "https://www.dropbox.com/s/m41kahhhx0a5z0u/data.tar.gz?dl=1" -O data.tar.gz
tar -xzf data.tar.gz
```

The current `train.py` expects dataset files in the format used by the Book / Taobao experiments:

```text
<dataset>_train.txt
<dataset>_valid.txt
<dataset>_test.txt
<dataset>_item_cate.txt
```

## Running the Code

Install dependencies from the repository's dependency file:

```bash
pip install -r requirements
```

The command-line entry point is `train.py`.

### Train

```bash
python train.py -p train --dataset book --model_type ComiRec-SA
```

### Test

```bash
python train.py -p test --dataset book --model_type ComiRec-SA
```

### Export embeddings

```bash
python train.py -p output --dataset book --model_type ComiRec-SA
```

Supported model names include:

```text
DNN
GRU4REC
MIND
ComiRec-DR
ComiRec-SA
```

### Important portability note

The current training script still contains local dataset-path defaults for the Book and Taobao datasets. Before running on a new machine, update the dataset path configuration in `train.py` or adapt it to your local directory layout.

This is a known engineering limitation of the current reproduction code and is intentionally documented rather than hidden.

## Evaluation

The evaluation pipeline retrieves Top-K items from learned item embeddings and reports metrics such as:

```text
Recall@K
NDCG@K
HitRate@K
Diversity@K
```

FAISS is used when available; otherwise the code falls back to NumPy similarity search.

## What This Repository Demonstrates

From an engineering / research-training perspective, the project demonstrates:

- sequential recommendation modeling;
- user and item embeddings;
- multi-interest representation learning;
- capsule-network dynamic routing;
- self-attention interest extraction;
- retrieval with FAISS;
- ranking evaluation;
- reproduction and adaptation of recommendation-system research code.

## Research Integrity / Attribution

This codebase is maintained as a **study and reproduction repository**. The underlying multi-interest modeling ideas originate from prior recommendation-system research, including the KDD 2020 work *Controllable Multi-Interest Framework for Recommendation*.

No claim is made here that the baseline architectures in this repository are novel contributions of this repository.

For graduate-application review, this repository should be read as evidence of recommendation-model implementation and experimentation, while separate research projects should be evaluated from their own codebases and documentation.