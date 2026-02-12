"""Exercise 10: Hierarchical clustering on generated points."""

# Step 1: Import required tools.
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

# Step 2: Create generated sample data with 4 groups.
input_points, _ = make_blobs(n_samples=100, centers=4, random_state=42)

# Step 3: Create hierarchical clustering model for 4 clusters.
model = AgglomerativeClustering(n_clusters=4)

# Step 4: Train model and get cluster labels.
cluster_labels = model.fit_predict(input_points)

# Step 5: Print first 20 labels.
# `[:20]` means first 20 values.
print("Cluster labels (first 20):", cluster_labels[:20])
