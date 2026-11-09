import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.generative.diffusion import DiffusionModel
print('Loaded', 'DiffusionModel')
