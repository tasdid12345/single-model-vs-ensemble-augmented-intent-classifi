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

## Results (latest run)

On the CLINC150 in-scope test split (10 labeled queries per intent, 3 seeds), a CV-tuned word+character TF-IDF logistic regression reached 82.3% test accuracy versus 79.1% for a soft-voting ensemble of classical classifiers, a gain of about 3.2 points with seed std near 0.8.

| Metric | Value |
|---|---|
| baseline_score | 0.791 |
| improved_score | 0.8232 |
| baseline_score_std | 0.006872 |
| improved_score_std | 0.007771 |
| improvement | 0.03222 |
| baseline_f1 | 0.7878 |
| improved_f1 | 0.8203 |
| baseline_f1_std | 0.006885 |
| improved_f1_std | 0.008213 |
| baseline_train_seconds | 6.233 |
| improved_train_seconds | 38.56 |

Run on 2026-10-11. Full output: [results.json](results.json) · [run.log](run.log)

Notes: T5/transformers and human augmentation replaced by simple random perturbation; baseline is not the paper's method. n_train is the number of original labeled queries (augmented set is 5x larger). Official val split unused; the official test split is used once per model. Baseline uses word features only and fixed hyper-parameters, so feature set and tuning both differ from the improved model. Selected C per seed: [1, 10, 10]. Std is over 3 seeds, which change the subsample and augmentation. SVM calibration inside the ensemble uses internal CV that is not group-aware.

## Reference

This project is motivated by the following paper:

> JJ Bird, A Ekárt, DR Faria (2023). *Chatbot Interaction with Artificial Intelligence: human data augmentation with T5 and language transformer ensemble for text classification*. https://link.springer.com/article/10.1007/s12652-021-03439-8

Paper author: Prof. Anikó Ekárt, Aston University (a.ekart@aston.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
