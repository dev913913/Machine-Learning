"""Exercise 7: k-NN classification for the iris dataset with correct/incorrect outputs."""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main() -> None:
    # Load the iris dataset.
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )

    # Create and train the k-NN model.
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Predict on the test set.
    predictions = model.predict(X_test)

    # Print correct and wrong predictions.
    for features, predicted, actual in zip(X_test, predictions, y_test):
        status = "correct" if predicted == actual else "wrong"
        print(f"{status:7} | predicted={predicted} actual={actual} features={features}")


if __name__ == "__main__":
    main()
