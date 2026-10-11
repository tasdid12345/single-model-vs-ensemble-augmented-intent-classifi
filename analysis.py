"""CLINC150 few-shot intent classification: CV-tuned word+char TF-IDF LR vs soft-voting ensemble."""
import json, os, random, sys, time
import numpy as np, pandas as pd, requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_small.json"
SEEDS, PER_INTENT, N_AUG = [42, 43, 44], 10, 4


def download(url):
    """GET with timeout and exponential backoff on 429/5xx."""
    for i in range(6):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429,) and r.status_code < 500:
                r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2 ** i)
    sys.exit("download failed")


def perturb(text, rng):
    """Random word dropout (p=0.1) and one random adjacent-word swap."""
    w = [t for t in text.split() if rng.random() > 0.1] or text.split()
    if len(w) > 1:
        i = rng.randrange(len(w) - 1)
        w[i], w[i + 1] = w[i + 1], w[i]
    return " ".join(w)


def build_train(train, seed):
    """Subsample PER_INTENT queries per intent, then add N_AUG perturbed copies each."""
    rng = random.Random(seed)
    by = {}
    for t, l in train:
        by.setdefault(l, []).append(t)
    X, y, g = [], [], []
    for l in sorted(by):
        for t in rng.sample(by[l], PER_INTENT):
            for k in range(N_AUG + 1):  # copy 0 = original
                X.append(t if k == 0 else perturb(t, rng)); y.append(l); g.append(len(g) // (N_AUG + 1))
    return np.array(X), np.array(y), np.array(g)


def baseline(seed):
    """Soft-voting ensemble (LR, calibrated linear SVM, complement NB) on word TF-IDF."""
    vote = VotingClassifier([
        ("lr", LogisticRegression(C=10, max_iter=300, random_state=seed)),
        ("svm", CalibratedClassifierCV(LinearSVC(random_state=seed), cv=3)),
        ("cnb", ComplementNB(alpha=0.1))], voting="soft", n_jobs=-1)
    return Pipeline([("tf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)), ("clf", vote)])


def improved(seed, cv):
    """Single LR on word+char TF-IDF; C chosen by group-aware CV on training data only."""
    feats = FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2))])
    pipe = Pipeline([("tf", feats), ("clf", LogisticRegression(max_iter=300, random_state=seed))])
    return GridSearchCV(pipe, {"clf__C": [1, 10, 100]}, cv=cv, scoring="accuracy", n_jobs=-1)


def evaluate(model, Xte, yte):
    p = model.predict(Xte)
    return accuracy_score(yte, p), f1_score(yte, p, average="macro")


data = download(URL)
train, test = data["train"], data["test"]  # in-scope only (oos_* keys ignored)
n_rows = sum(len(data[k]) for k in ("train", "val", "test"))
Xte, yte = np.array([t for t, _ in test]), np.array([l for _, l in test])
print(f"in-scope rows {n_rows}, official test {len(test)}")

res = {"base": [], "imp": []}
best_C = []
for seed in SEEDS:
    np.random.seed(seed)
    Xtr, ytr, gtr = build_train(train, seed)
    # CV folds keep an original query and its augmented copies together (no leakage)
    cv = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr, gtr))
    t0 = time.time(); b = baseline(seed).fit(Xtr, ytr); tb = time.time() - t0
    t0 = time.time(); m = improved(seed, cv).fit(Xtr, ytr); ti = time.time() - t0
    best_C.append(m.best_params_["clf__C"])
    ab, fb = evaluate(b, Xte, yte); ai, fi = evaluate(m, Xte, yte)
    res["base"].append((ab, fb, tb)); res["imp"].append((ai, fi, ti))
    print(f"seed {seed}: baseline acc {ab:.4f} ({tb:.0f}s) | improved acc {ai:.4f} C={best_C[-1]} ({ti:.0f}s)")

B, I = np.array(res["base"]), np.array(res["imp"])
if not np.isfinite(B).all() or not np.isfinite(I).all() or B[:, 0].mean() < 0.1:
    sys.exit("metrics not meaningful")
metrics = {
    "baseline_score": B[:, 0].mean(), "improved_score": I[:, 0].mean(),
    "baseline_score_std": B[:, 0].std(), "improved_score_std": I[:, 0].std(),
    "improvement": I[:, 0].mean() - B[:, 0].mean(),
    "baseline_f1": B[:, 1].mean(), "improved_f1": I[:, 1].mean(),
    "baseline_f1_std": B[:, 1].std(), "improved_f1_std": I[:, 1].std(),
    "baseline_train_seconds": B[:, 2].mean(), "improved_train_seconds": I[:, 2].mean(),
}
metrics = {k: float(v) for k, v in metrics.items()}
print(json.dumps(metrics, indent=2))

os.makedirs("figures", exist_ok=True)
plt.bar(["Soft-voting ensemble", "Word+char LR (CV-tuned)"], [metrics["baseline_score"], metrics["improved_score"]],
        yerr=[metrics["baseline_score_std"], metrics["improved_score_std"]], color=["gray", "tab:blue"])
plt.ylabel("Test accuracy"); plt.title("CLINC150, 10 labeled queries/intent"); plt.tight_layout()
plt.savefig("figures/accuracy.png", dpi=120)

out = {
    "question": "With scarce labeled chatbot text and the same augmentation, can one CV-tuned word+character TF-IDF logistic-regression model match or beat a soft-voting ensemble of several classifiers on intent accuracy?",
    "method": f"CLINC150 in-scope queries (data_small.json); {PER_INTENT} labeled training queries per intent subsampled from the official train split, each plus {N_AUG} random word dropout/swap copies; evaluated on the official in-scope test split; 3 seeds (subsample + augmentation).",
    "dataset": {"name": "CLINC150 (data_small, in-scope only)", "source_url": URL, "n_rows": int(n_rows),
                "n_train": int(len(set(ytr)) * PER_INTENT), "n_test": int(len(test)),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "test accuracy", "higher_is_better": True,
    "baseline": "Soft-voting ensemble of logistic regression, calibrated linear SVM and complement naive Bayes on word (1-2gram) TF-IDF, trained on original + augmented queries (stand-in for the paper's transformer ensemble).",
    "improved": "Single logistic regression on word + character TF-IDF with C in {1,10,100} chosen by 3-fold group-aware CV on the training set, same augmented data.",
    "metrics": metrics,
    "notes": (f"T5/transformers and human augmentation replaced by simple random perturbation; baseline is not the paper's method. "
              f"n_train is the number of original labeled queries (augmented set is {(N_AUG + 1)}x larger). Official val split unused; the official test split is used once per model. "
              f"Baseline uses word features only and fixed hyper-parameters, so feature set and tuning both differ from the improved model. "
              f"Selected C per seed: {best_C}. Std is over 3 seeds, which change the subsample and augmentation. "
              f"SVM calibration inside the ensemble uses internal CV that is not group-aware."),
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("wrote results.json")
