"""Exercise 4: Gaussian Naive Bayes classifier using a CSV dataset."""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main() -> None:
    # Load numeric data from CSV.
    data = pd.read_csv("data/naive_bayes_train.csv")

    # Features are the input columns.
    X = data[["feature_1", "feature_2"]]
    # Label is the output column.
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Create and train the Naive Bayes model.
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predict on the test set.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Predictions:", predictions)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
