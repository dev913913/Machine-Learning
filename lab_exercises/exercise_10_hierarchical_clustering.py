"""Exercise 10: Hierarchical clustering on synthetic data."""

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs


def main() -> None:
    # Create synthetic data with 4 centers.
    X, _ = make_blobs(n_samples=100, centers=4, random_state=42)

    # Create and fit the hierarchical clustering model.
    model = AgglomerativeClustering(n_clusters=4, linkage="ward")
    labels = model.fit_predict(X)

    # Print a few labels to see the result.
    print("Cluster labels (first 20):", labels[:20])


if __name__ == "__main__":
    main()
