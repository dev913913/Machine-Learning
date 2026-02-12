"""Exercise 8: K-means clustering on synthetic data."""

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def main() -> None:
    # Create synthetic data with 3 centers.
    X, _ = make_blobs(n_samples=150, centers=3, random_state=42)

    # Create and fit the K-means model.
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    # Print the cluster centers and some labels.
    print("Cluster centers:\n", model.cluster_centers_)
    print("First 10 labels:", labels[:10])


if __name__ == "__main__":
    main()
