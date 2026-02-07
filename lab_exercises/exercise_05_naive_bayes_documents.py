"""Exercise 5: Document classification with Naive Bayes."""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def main() -> None:
    # Example documents (short texts).
    documents = [
        "apple orange banana",
        "banana apple fruit",
        "dog cat hamster",
        "cat dog pet",
        "car bus train",
        "train car transport",
    ]
    # Labels for each document.
    labels = ["fruit", "fruit", "animal", "animal", "vehicle", "vehicle"]

    # Convert text into word-count vectors.
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.33, random_state=42, stratify=labels
    )

    # Create and train the Naive Bayes model for text data.
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predict on the test set and compute metrics.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro")
    recall = recall_score(y_test, predictions, average="macro")

    print("Predictions:", predictions)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")


if __name__ == "__main__":
    main()
