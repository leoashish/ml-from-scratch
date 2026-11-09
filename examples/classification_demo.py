"""Demo: classification algorithms."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data import make_classification, make_moons
from src.utils.preprocessing import train_test_split
from src.utils.metrics import accuracy
from src.classification.logistic_regression import LogisticRegression
from src.classification.svm import LinearSVM, KernelSVM

X, y = make_classification(n_samples=400, n_features=2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)

for name, model in [("LogisticRegression", LogisticRegression(lr=0.1, n_iters=500)),
                    ("LinearSVM", LinearSVM(C=1.0, lr=0.01, n_iters=300)),
                    ("KernelSVM-RBF", KernelSVM(C=1.0, kernel="rbf", max_passes=5))]:
    model.fit(Xtr, ytr)
    print(f"{name:18s} acc={accuracy(yte, model.predict(Xte)):.4f}")

# multiclass
X, y = make_classification(n_samples=300, n_features=2, n_classes=3, random_state=1)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=1)
lr = LogisticRegression(lr=0.1, n_iters=800)
lr.fit(Xtr, ytr)
print("Multiclass LR acc=", round(accuracy(yte, lr.predict(Xte)), 4))
