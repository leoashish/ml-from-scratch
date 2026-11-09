"""Demo: clustering algorithms."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.utils.data import make_blobs
from src.utils.metrics import silhouette_score
from src.clustering.kmeans import KMeans
from src.clustering.dbscan import DBSCAN

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

km = KMeans(n_clusters=4, random_state=0)
km.fit(X)
print("KMeans: inertia=%.2f  silhouette=%.4f  clusters=%s" %
      (km.inertia_, silhouette_score(X, km.labels_), np.unique(km.labels_, return_counts=True)[1]))

db = DBSCAN(eps=1.0, min_samples=5)
db.fit(X)
print("DBSCAN: labels=%s" % (np.unique(db.labels_, return_counts=True),))
