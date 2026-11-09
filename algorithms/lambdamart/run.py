import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ranking.lambdamart import LambdaMART
print('Loaded', 'LambdaMART')
