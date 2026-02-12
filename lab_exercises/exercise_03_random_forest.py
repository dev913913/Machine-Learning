"""Exercise 3: Random Forest classification using the iris dataset."""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main() -> None:
    # Load the iris dataset.
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
    )

    # Create and train the Random Forest model.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Report accuracy on the test set.
    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
