# Sexism Detection with RNNs and Transformers

This repository contains the solution for **Assignment 1** of the NLP course. The project addresses **Task 2 of the EXIST 2023 Benchmark**, focusing on the multi-class classification of sexism in social media text.

**Credits:** Federico Ruggeri, Eleonora Mancini, Paolo Torroni

## 👥 Authors
* **Omid Nejati**
* **Alireza Shahidiani**

##  Project Overview

The goal is to classify English tweets into one of four categories based on the author's intention regarding sexism. Unlike simple binary classification, this task distinguishes between direct sexism, reporting sexism, and judging sexism.

### The Task (EXIST 2023 Task 2)
Given a tweet, the model predicts one of the following labels:
* **0: Non-sexist** (`-`)
* **1: DIRECT**: The message itself is sexist or incites sexism.
* **2: JUDGEMENTAL**: The message describes sexist behavior to condemn it.
* **3: REPORTED**: The message reports a sexist situation experienced by the author or others.

##  Methodology

The project explores two main deep learning approaches:

### 1. Custom RNN Models (TensorFlow/Keras)
* **Preprocessing:** Text cleaning (removing URLs, emojis, hashtags), lemmatization using NLTK.
* **Embeddings:** Uses pre-trained **GloVe 50d** word vectors. OOV tokens are initialized randomly.
* **Architectures:**
    * **Baseline:** A single Bidirectional LSTM layer followed by a Dense classifier.
    * **Stacked:** Two stacked Bidirectional LSTM layers for capturing more complex patterns.

### 2. Transformer Model (Hugging Face)
* **Model:** Fine-tuning **`cardiffnlp/twitter-roberta-base-hate`**, a RoBERTa model pre-trained on tweets.
* **Training:** Uses the Hugging Face `Trainer` API with early stopping and optimal hyperparameter selection.

##  Dataset

The project uses the **EXIST 2023** dataset (Learning with Disagreement), filtered to include only **English** samples.
* **Labels:** Aggregated from 6 annotators using **Majority Voting**. Samples with no clear majority were discarded.
* **Class Imbalance:** The dataset is heavily imbalanced, with "Non-sexist" being the majority class (~73%).

##  Results

Models were evaluated using **Macro F1-Score**, Precision, and Recall on a held-out test set.

| Model | Macro F1 | Macro Precision | Macro Recall |
| :--- | :--- | :--- | :--- |
| **Baseline (BiLSTM)** | 0.4330 | 0.5849 | 0.4176 |
| **Transformer (RoBERTa)** | **0.5234** | 0.5036 | 0.5496 |

*Key Findings:*
* The **Transformer model outperformed the RNNs** significantly (+9% F1), demonstrating the power of pre-training on domain-specific data (Twitter).
* The LSTM models struggled significantly with the minority classes (`JUDGEMENTAL` and `REPORTED`), often failing to predict them entirely due to the small sample size.

##  Installation & Usage

**1. Prerequisites**
* Python 3.8+
* GPU support recommended (Google Colab T4 used for experiments)

**2. Install Dependencies**
```bash
pip install tensorflow transformers datasets scikit-learn pandas matplotlib seaborn nltk
```
## Project Structure

* Data Processing: JSON parsing, Label Encoding, Cleaning pipeline.

* Task 1-3: Corpus analysis, Cleaning, and Vectorization (GloVe).

* Task 4-5: RNN Model definition, training loop with multiple seeds, and selection of the best model.

* Task 6: Transformer fine-tuning pipeline.

* Task 7: Comprehensive Error Analysis (Confusion Matrices, OOV analysis, False Positives/Negatives).
## License
This project is created for academic purposes. The dataset is part of the [EXIST 2023 Challenge](https://clef2023.clef-initiative.eu/index.php?page=Pages/labs.html#EXIST).
