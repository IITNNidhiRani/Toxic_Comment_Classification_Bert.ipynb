
# Toxic Comment Classification with BERT

This repository implements a multi‐label toxic comment classifier based on fine‐tuning BERT.  
Given an input comment, the model predicts six toxicity subtypes: **toxic**, **severe toxic**, **obscene**, **threat**, **insult**, and **identity hate**.

---

##  Table of Contents

1. [Background](#background)  
2. [Features](#features)  
3. [Data](#data)  
4. [Installation & Requirements](#installation--requirements)  
5. [Usage](#usage)  
6. [Notebook Overview](#notebook-overview)  
7. [Results](#results)  
8. [Citations](#citations)  

---

## Background

Automated moderation systems must contend with subtle, context-dependent toxicity in text.  
This notebook fine‐tunes a pre-trained BERT (bert-base-uncased) model on the **Wikipedia Toxicity Subtypes** dataset, which contains ∼127 K comments annotated for six toxicity labels.

---

##  System Architecture

![System Architecture of UniFairNet](images/architecture.png)

UniFairNet is a modular pipeline for multimodal hateful content detection and blurring:

### 1. Data & Preprocessing  
1. **Load Dataset**  
   - Read `train.csv`, `dev.csv`, `test.csv`, etc.  
   - Contains `id`, `comment_text`, and six binary toxicity labels.  
2. **Tokenization**  
   - Use `BertTokenizer` (`bert-base-uncased`) with `max_length=128`.  
   - Produce `input_ids` & `attention_mask`.  
3. **Dataset Creation**  
   - Wrap into PyTorch `Dataset`/`DataLoader` for train/val/test.  

### 2. Model Setup & Training  
1. **Model Definition**  
   - `BertForSequenceClassification.from_pretrained(..., num_labels=6)`  
   - Sigmoid‐activated head for multi‐label outputs.  
2. **Optimizer & Scheduler**  
   - `AdamW(lr=2e-5)`, optional LR scheduler with warm-up.  
3. **Training Loop**  
   - 3 epochs, BCE loss, log loss every 100 steps (`training_log.xlsx`).  
4. **Save Artifacts**  
   - `model.save_pretrained("model/")`  
   - `tokenizer.save_pretrained("model/")`  

### 3. Inference & Evaluation  
1. **Inference**  
   - Reload model/tokenizer in `eval()` mode.  
   - Run test split → sigmoid‐probs → threshold at 0.5 → `test_predicted.csv`.  
2. **Evaluation Metrics**  
   - **Accuracy**: exact‐match across all six labels.  
   - **Macro‐F1** and **Macro‐ROC‐AUC** averaged over labels.  
 
##  Features

- **Multi‐label classification** over six toxicity subtypes  
- **Data preprocessing** with the BERT tokenizer (max length 128)  
- **Fine-tuning** using AdamW, binary cross-entropy loss, 3 epochs, batch size 32, LR = 2×10⁻⁵  
- **Training dynamics logging** (loss every 100 steps → `training_log.xlsx`)  
- **Inference & export** of per‐comment probabilities and binary predictions to `test_predicted.csv`  
- **Evaluation metrics**: per‐label ROC-AUC, macro-averaged F1, overall accuracy  

---

## Data

We use the **Wikipedia Toxicity Subtypes** split as follows:

| Split       | Samples  |
| ----------- | -------- |
| Training    | 102 256  |
| Validation  | 12 564   |
| Test        | 12 564   |
| Held-out    | 999      |

Download `train.csv`, `dev.csv`, `test.csv`, and `test_public_expanded.csv` from the [Kaggle Jigsaw Toxic Comment challenge](https://kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) and place them in a `data/` folder.

---



📓 Notebook Overview
Setup – imports, device check

Data Loading – CSV → datasets

Preprocessing – BERT tokenization

Model Definition – BertForSequenceClassification with 6 outputs

Training Loop – AdamW, BCE loss, logging

Validation – compute F1 & ROC-AUC per epoch

Test Inference – export predictions to CSV

Metrics & Analysis – final Accuracy, Macro-F1, ROC-AUC

Gradio Demo (optional)

📈 Results
Test Accuracy: 0.9123

Macro-F1: 0.9075

Macro-ROC-AUC: 0.9621

Citations
If you use this work, please cite:

N. Rani, “Fine-Tuning Large Language Models Using NLP Techniques for Enhanced Bias Mitigation in Generative AI,” M.Tech project report, IIT Jodhpur, 2025.

And for the dataset:

1. Wikipedia Toxicity Subtypes (Text)
• Comments from Wikipedia annotated for toxicity types.
• Human-labeled, split: ~80% train, 10% val, 10% test.
•https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
