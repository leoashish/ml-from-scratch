import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.clustering.kmeans import KMeans
print('Loaded', 'KMeans')
