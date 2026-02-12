"""Exercise 8: K-means clustering on generated points."""

# Step 1: Import required tools.
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Step 2: Create artificial data with 3 groups.
# `_` is a common name for a value we do not plan to use.
input_points, _ = make_blobs(n_samples=150, centers=3, random_state=42)

# Step 3: Create K-means model for 3 clusters.
model = KMeans(n_clusters=3, random_state=42, n_init=10)

# Step 4: Train model and get cluster label for each point.
cluster_labels = model.fit_predict(input_points)

# Step 5: Print cluster centers.
print("Cluster centers:\n", model.cluster_centers_)

# Step 6: Print first 10 labels.
# `[:10]` means from start up to index 10 (not including 10).
print("First 10 labels:", cluster_labels[:10])
