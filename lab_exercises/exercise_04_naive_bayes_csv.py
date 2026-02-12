"""Exercise 4: Gaussian Naive Bayes using numeric values from CSV."""

# Step 1: Import required tools.
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Step 2: Read data from CSV file.
data_table = pd.read_csv("data/naive_bayes_train.csv")

# Step 3: Select input columns.
# Double brackets keep the result as a table (DataFrame).
input_values = data_table[["feature_1", "feature_2"]]

# Step 4: Select output column.
output_values = data_table["label"]

# Step 5: Split data into training and test sets.
train_input, test_input, train_output, test_output = train_test_split(
    input_values, output_values, test_size=0.25, random_state=42
)

# Step 6: Create model and train it.
model = GaussianNB()
model.fit(train_input, train_output)

# Step 7: Predict output labels for test data.
predicted_output = model.predict(test_input)

# Step 8: Compute accuracy.
accuracy = accuracy_score(test_output, predicted_output)

# Step 9: Print results.
print("Predictions:", predicted_output)
print("Accuracy:", round(accuracy, 2))
