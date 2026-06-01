# Text Mining Pipeline — NER, Sentiment Analysis & Topic Classification

Group 19 | Vrije Universiteit Amsterdam | Text Mining Course 2026  
**Authors:** Minhye Kang, Sebastian Nagler, Boglarka Ori, Louis Battle

---

## Overview

This repository contains three NLP pipelines evaluated on a shared test set:

| Task | Notebook | Model |
|------|----------|-------|
| NERC — System A | `NERC_1st_system.ipynb` | `spaCy en_core_web_sm` |
| NERC — System B | `NERC_2nd_system.ipynb` | `dslim/bert-base-NER` |
| Sentiment Analysis | `sentiment-analysis.ipynb` | `cardiffnlp/twitter-roberta-base-sentiment` |
| Topic Classification | `topic_analysis.ipynb` | `bert-base-cased` (SimpleTransformers) |

---

## Repository Structure

```
TM-SENTIMENT-ANALYSIS/
├── data/
│   └── raw/
│       ├── train/
│       │   └── airlinetweets/
│       │       ├── negative/       ← one .txt file per tweet
│       │       ├── neutral/
│       │       └── positive/
│       └── test/
│           └── *.tsv               ← Sentiment-topic-test.tsv
├── results/                        ← model checkpoints + plots (gitignored)
├── src/
│   └── train.py                    ← standalone sentiment training script
├── NERC_1st_system.ipynb           ← NERC with spaCy
├── NERC_2nd_system.ipynb           ← NERC with BERT
├── sentiment-analysis.ipynb        ← Sentiment Analysis with RoBERTa
├── topic_analysis.ipynb            ← Topic Classification with BERT
├── NER-test.tsv                    ← NER test set (place in root)
├── Sentiment-topic-test.tsv        ← Sentiment/topic test set (place in root)
├── requirements.txt
└── README.md
```

---

## Test Sets

Download from VU Canvas and place in the project root:
- `NER-test.tsv` — columns: `sentence id`, `token id`, `token`, `BIO NER tag`
- `Sentiment-topic-test.tsv` — columns: `id`, `text`, `sentiment`, `topic`

---

## Setup

**Requirements:** Python 3.9+

```bash
git clone https://github.com/l-battle/tm-sentiment-analysis
cd tm-sentiment-analysis
pip install -r requirements.txt
```

---

## 1. NERC — System A (spaCy)

**Notebook:** `NERC_1st_system.ipynb`

**Install:**
```bash
pip install "numpy<2" spacy
python -m spacy download en_core_web_sm
```

**What it does:**
- Loads `NER-test.tsv` from the project root
- Runs `en_core_web_sm` on each sentence and converts predictions to BIO tags
- Evaluates against gold labels using `classification_report`
- Saves predictions to `NER_spaCy_predictions.csv`

**Run:** Open the notebook and run all cells top to bottom.  
`NER-test.tsv` must be in the same directory as the notebook.

**Expected result:** weighted F1 = 0.86

---

## 2. NERC — System B (BERT)

**Notebook:** `NERC_2nd_system.ipynb`

**Install:**
```bash
pip install transformers torch
```

**What it does:**
- Downloads `dslim/bert-base-NER` from HuggingFace (~400MB, cached after first run)
- Runs token classification on each sentence in `NER-test.tsv`
- Evaluates against gold labels using `classification_report`
- Saves predictions to `NER_BERT_predictions.csv`

**Run:** Open the notebook and run all cells top to bottom.  
`NER-test.tsv` must be in the same directory as the notebook.

**Expected result:** weighted F1 = 0.98, macro F1 = 0.92

---

## 3. Sentiment Analysis (RoBERTa)

**Notebook:** `sentiment-analysis.ipynb`  
**Script:** `src/train.py`

**Install:**
```bash
pip install torch transformers scikit-learn matplotlib seaborn
```

For GPU support (recommended), install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> **Windows users:** set `num_workers` to `0` in `CONFIG` — the notebook handles this automatically via `os.name` check.

**What it does:**
- Loads airline tweets from `data/raw/train/airlinetweets/{negative,neutral,positive}/`
- Loads test data from `data/raw/test/Sentiment-topic-test.tsv`
- Fine-tunes `cardiffnlp/twitter-roberta-base-sentiment` for 4 epochs
- Saves best checkpoint to `results/` based on validation accuracy
- Exports `results/confusion_matrix.png` and `results/training_curves.png`

**Run from project root:**
```bash
# Notebook
jupyter notebook sentiment-analysis.ipynb

# Or script
python src/train.py
```

**Expected result:** weighted F1 = 0.80, accuracy = 0.80

---

## 4. Topic Classification (BERT)

**Notebook:** `topic_analysis.ipynb`

**Install:**
```bash
pip install simpletransformers==0.70.5
```

> Restart your runtime/kernel after installing simpletransformers.

**Training data:** Download manually from Kaggle and place CSV files in the project root or update the paths in the notebook:
- [IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) → `IMDB Dataset.csv`
- [Yelp Restaurant Reviews](https://www.kaggle.com/datasets/farukalam/yelp-restaurant-reviews) → `yelp.csv`
- [Amazon Book Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) → `books_data.csv`

**What it does:**
- Truncates each dataset to 1000 samples and assigns a class label (0=Movie, 1=Restaurant, 2=Book)
- Concatenates, shuffles, and splits 90/10 train/dev
- Fine-tunes `bert-base-cased` via SimpleTransformers (lr=4e-6)
- Evaluates on the 10 hardcoded test sentences from `Sentiment-topic-test.tsv`

**Run:** Open the notebook and run all cells top to bottom.

> **Note:** Topic classification was developed on Google Colab with GPU. Training locally on CPU will be significantly slower (~1–2 hours). Using a GPU is strongly recommended.

**Expected result:** F1 = 1.0 on all classes

---

## Results Summary

| Task | System | Weighted F1 |
|------|--------|-------------|
| NERC | spaCy (System A) | 0.86 |
| NERC | BERT (System B) | 0.98 |
| Sentiment Analysis | twitter-roberta | 0.80 |
| Topic Classification | bert-base-cased | 1.00 |

---

## References

- Devlin et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*, 4171–4186.
- Barbieri et al. (2022). XLM-T: Multilingual language models in Twitter for sentiment analysis and beyond. *LREC 2022*.
- Sun et al. (2019). How to fine-tune BERT for text classification? *arXiv:1905.05583*.
