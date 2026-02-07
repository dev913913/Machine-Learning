"""Exercise 2: Decision Tree classification using a simple weather dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    # Load the dataset from CSV.
    data = pd.read_csv("data/decision_tree_data.csv")

    # Convert text columns to numbers using one-hot encoding.
    X = pd.get_dummies(data.drop(columns=["play"]))
    # Target column (the label we want to predict).
    y = data["play"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Create and train the Decision Tree model.
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Check accuracy on the test set.
    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.2f}")

    # Try a new sample and predict the label.
    new_sample = pd.DataFrame(
        [
            {
                "outlook": "sunny",
                "temperature": "cool",
                "humidity": "high",
                "windy": False,
            }
        ]
    )
    # Align the new sample's columns with the training columns.
    new_sample_encoded = pd.get_dummies(new_sample).reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(new_sample_encoded)[0]
    print("New sample prediction:", prediction)


if __name__ == "__main__":
    main()
