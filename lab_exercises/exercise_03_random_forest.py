"""Exercise 3: Random Forest classification with the iris dataset."""

# Step 1: Import required tools.
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Step 2: Load iris dataset.
iris_data = load_iris()

# Step 3: Split into training and test data.
train_input, test_input, train_output, test_output = train_test_split(
    iris_data.data, iris_data.target, test_size=0.25, random_state=42
)

# Step 4: Create Random Forest model.
# `n_estimators=100` means use 100 decision trees.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 5: Train model.
model.fit(train_input, train_output)

# Step 6: Check accuracy on test data.
accuracy = model.score(test_input, test_output)
print("Test accuracy:", round(accuracy, 2))
