import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.forecasting.arima import ARIMA
print('Loaded', 'ARIMA')
