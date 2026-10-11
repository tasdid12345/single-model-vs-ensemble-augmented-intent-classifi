# Few-shot intent classification: tuned TF-IDF logistic regression vs. soft-voting ensemble

## Motivation
The paper "Chatbot Interaction with Artificial Intelligence: human data augmentation with T5 and language transformer ensemble for text classification" (2023) combines data augmentation with an ensemble of language transformers to classify chatbot text when labeled data is scarce. Transformers and T5 cannot run on CPU here, so this experiment asks whether a much simpler single model can compete with an ensemble of classical classifiers under the same augmentation.

## Question
With scarce labeled chatbot text and the same augmentation, can one CV-tuned word+character TF-IDF logistic-regression model match or beat a soft-voting ensemble of several classifiers on intent accuracy?

## Dataset
CLINC150 `data_small.json` (clinc/oos-eval GitHub repo), in-scope queries only. 10 labeled training queries per intent are subsampled from the official train split; evaluation uses the official in-scope test split. This is **not** the paper's own dataset.

## Models
- **Baseline:** soft-voting ensemble (logistic regression, calibrated linear SVM, complement naive Bayes) on word TF-IDF. It stands in for the paper's transformer ensemble and is not the paper's method.
- **Improved:** a single logistic regression on word + character TF-IDF, with C chosen by group-aware cross-validation on the training set only.
- Both are trained on the same original plus randomly perturbed (word dropout/swap, 4 copies per query) data, over seeds 42, 43 and 44.
- Metrics: test accuracy (main), macro-F1 and training time (secondary).

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> JJ Bird, A Ekárt, DR Faria (2023). *Chatbot Interaction with Artificial Intelligence: human data augmentation with T5 and language transformer ensemble for text classification*. https://link.springer.com/article/10.1007/s12652-021-03439-8

Paper author: Prof. Anikó Ekárt, Aston University (a.ekart@aston.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
