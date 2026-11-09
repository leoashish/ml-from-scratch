import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.classification.logistic_regression import LogisticRegression
print('Loaded', 'LogisticRegression')
