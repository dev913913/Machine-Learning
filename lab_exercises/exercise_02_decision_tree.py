"""Exercise 2: Decision Tree classification with a simple weather dataset."""

# Step 1: Import required tools.
# `from ... import ...` means import specific names from a package.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Step 2: Load the CSV dataset.
data_table = pd.read_csv("data/decision_tree_data.csv")

# Step 3: Separate input columns from output column.
# Input columns are all columns except "play".
input_table = data_table.drop(columns=["play"])
# Output column is "play" (what we want to predict).
output_values = data_table["play"]

# Step 4: Convert text categories into numeric columns.
# Machine learning models usually need numbers, not text labels.
input_encoded = pd.get_dummies(input_table)

# Step 5: Split data into training set and test set.
# The model learns from training data and is checked on test data.
train_input, test_input, train_output, test_output = train_test_split(
    input_encoded, output_values, test_size=0.25, random_state=42
)

# Step 6: Create a Decision Tree model.
# `random_state=42` keeps results repeatable.
model = DecisionTreeClassifier(random_state=42)

# Step 7: Train the model.
# `fit` means learn patterns from input and output examples.
model.fit(train_input, train_output)

# Step 8: Evaluate model accuracy.
# `score` returns fraction of correct predictions.
accuracy = model.score(test_input, test_output)
print("Test accuracy:", round(accuracy, 2))

# Step 9: Create one new weather sample.
# `{}` makes a dictionary and `[]` makes a list.
new_sample = pd.DataFrame(
    [{"outlook": "sunny", "temperature": "cool", "humidity": "high", "windy": False}]
)

# Step 10: Convert new sample to same encoded format as training input.
# `reindex(..., fill_value=0)` adds any missing columns with value 0.
new_sample_encoded = pd.get_dummies(new_sample).reindex(columns=input_encoded.columns, fill_value=0)

# Step 11: Predict the output for the new sample.
# `[0]` means take the first prediction from the returned list/array.
prediction = model.predict(new_sample_encoded)[0]
print("New sample prediction:", prediction)
