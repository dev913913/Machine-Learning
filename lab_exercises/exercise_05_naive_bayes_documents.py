"""Exercise 5: Simple document classification with Naive Bayes."""

# Step 1: Import required tools.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Step 2: Create a small text dataset.
# Each string is one document.
documents = [
    "apple orange banana",
    "banana apple fruit",
    "dog cat hamster",
    "cat dog pet",
    "car bus train",
    "train car transport",
]

# Step 3: Create labels (correct class) for each document.
labels = ["fruit", "fruit", "animal", "animal", "vehicle", "vehicle"]

# Step 4: Convert words to numeric counts.
# ML model needs numeric input, so CountVectorizer turns text into numbers.
word_counter = CountVectorizer()
input_values = word_counter.fit_transform(documents)

# Step 5: Split into training and test data.
train_input, test_input, train_output, test_output = train_test_split(
    input_values, labels, test_size=0.33, random_state=42
)

# Step 6: Create Naive Bayes model and train it.
model = MultinomialNB()
model.fit(train_input, train_output)

# Step 7: Predict labels for test documents.
predicted_output = model.predict(test_input)

# Step 8: Compute accuracy.
accuracy = accuracy_score(test_output, predicted_output)

# Step 9: Print prediction results.
print("Predictions:", predicted_output)
print("Accuracy:", round(accuracy, 2))
