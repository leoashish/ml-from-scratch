import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.generative.autoregressive import AutoregressiveCharModel
print('Loaded', 'AutoregressiveCharModel')
