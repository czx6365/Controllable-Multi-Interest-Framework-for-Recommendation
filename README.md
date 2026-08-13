# Multi-Interest Recommendation Models

**A TensorFlow/Keras study implementation of sequential and multi-interest recommendation models.**

This repository is a **reproduction / learning implementation**, not a new recommendation paper. It is based on ideas from *Controllable Multi-Interest Framework for Recommendation* (KDD 2020) and related multi-interest recommendation work.

The purpose of the repository is to study how a single user behavior sequence can be represented by multiple latent interests and then used for retrieval-oriented recommendation.

> **Scope note for readers:** this repository is separate from my later work on controllable generative recommendation. It should not be interpreted as the codebase for discrete-diffusion / generative recommendation research.

## What Is Implemented

| Model | Main idea |
| --- | --- |
| `DNN` | pooled history embedding followed by feed-forward layers |
| `GRU4REC` | GRU-based sequential user representation |
| `MIND` | capsule-network-based multi-interest extraction |
| `ComiRec-DR` | dynamic-routing multi-interest representation |
| `ComiRec-SA` | self-attention-based multi-interest representation |

The model factory in [`model.py`](model.py) exposes these implementations through a shared training interface.

## System View

```text
User behavior sequence
        ↓
Sequence encoder
        ↓
Single / multiple interest embeddings
        ↓
FAISS or NumPy inner-product retrieval
        ↓
Top-K ranking
        ↓
Recall / NDCG / HitRate / Diversity
```

## Core Components

- [`model.py`](model.py): item embeddings, DNN/GRU encoders, capsule routing, self-attention interest extraction, and model save/load utilities.
- [`train.py`](train.py): portable training, retrieval, evaluation, early stopping, testing, and embedding export.
- [`data_iterator.py`](data_iterator.py): sequential recommendation data loading and history-mask construction.
- [`mostpop.py`](mostpop.py): popularity-oriented baseline.

## Data Layout

`train.py` no longer contains machine-specific Windows paths. Supply the dataset directory with `--data-dir`, the `REC_DATA_DIR` environment variable, or the default `data/<dataset>` layout.

Expected files:

```text
<data-dir>/
├── <dataset>_train.txt
├── <dataset>_valid.txt
├── <dataset>_test.txt
└── <dataset>_item_cate.txt
```

`item_count` is inferred from the item-category mapping by default and can be overridden with `--item-count`.

## Running the Code

Install dependencies:

```bash
pip install -r requirements
```

### Train

```bash
python train.py \
  -p train \
  --dataset book \
  --data-dir /path/to/book_data \
  --model-type ComiRec-SA \
  --experiment-name book_comirec_sa
```

Instead of passing the path each time:

```bash
export REC_DATA_DIR=/path/to/book_data
python train.py -p train --dataset book --model-type MIND
```

### Test

```bash
python train.py \
  -p test \
  --dataset book \
  --data-dir /path/to/book_data \
  --model-type ComiRec-SA \
  --experiment-name book_comirec_sa
```

### Export embeddings

```bash
python train.py \
  -p output \
  --dataset book \
  --data-dir /path/to/book_data \
  --model-type ComiRec-SA \
  --experiment-name book_comirec_sa
```

Useful overrides include `--batch-size`, `--maxlen`, `--test-iter`, `--item-count`, `--topN`, and `--coef`.

## Evaluation

The evaluation pipeline retrieves Top-K items from learned item embeddings and reports:

```text
Recall@K
NDCG@K
HitRate@K
Diversity@K
```

FAISS is used when available; otherwise the code falls back to NumPy inner-product search.

## Engineering Cleanup

The public training entry point is intentionally portable:

- no personal drive letters or local absolute paths;
- dataset directory configurable by CLI or environment variable;
- dataset files validated before training starts;
- item vocabulary size inferred when possible;
- dataset-specific hyperparameter defaults separated from filesystem configuration;
- non-interactive experiment naming for scripted runs.

## What This Repository Demonstrates

- sequential recommendation modeling;
- multi-interest representation learning;
- capsule-network dynamic routing;
- self-attention interest extraction;
- FAISS retrieval and ranking evaluation;
- reproducible configuration of research code;
- reproduction and adaptation of recommendation-system research.

## Research Integrity / Attribution

This codebase is maintained as a **study and reproduction repository**. The underlying multi-interest modeling ideas originate from prior recommendation-system research, including the KDD 2020 work *Controllable Multi-Interest Framework for Recommendation*.

No claim is made here that these baseline architectures are novel contributions of this repository. For graduate-application review, this repository should be read as evidence of recommendation-model implementation and experimentation; later generative-recommendation research is a separate project.
