import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.anomaly_detection.lof import LocalOutlierFactor
print('Loaded', 'LocalOutlierFactor')
