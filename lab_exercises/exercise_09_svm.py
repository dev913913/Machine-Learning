"""Exercise 9: Support Vector Machine classification."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main() -> None:
    # Load the breast cancer dataset.
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )

    # Scale the data so features are on the same range.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the SVM model.
    model = SVC(kernel="rbf", gamma="scale", random_state=42)
    model.fit(X_train_scaled, y_train)

    # Report accuracy on the test set.
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Test accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
