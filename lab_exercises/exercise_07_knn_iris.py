"""Exercise 7: k-Nearest Neighbors (k-NN) with the iris dataset."""

# Step 1: Import required tools.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Step 2: Load iris data.
iris_data = load_iris()

# Step 3: Split into training and test sets.
train_input, test_input, train_output, test_output = train_test_split(
    iris_data.data, iris_data.target, test_size=0.3, random_state=42
)

# Step 4: Create and train k-NN model.
# `n_neighbors=3` means model looks at 3 nearest points.
model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_input, train_output)

# Step 5: Predict classes for test rows.
predicted_output = model.predict(test_input)

# Step 6: Print each prediction and show if it is correct.
for one_input, one_prediction, real_output in zip(test_input, predicted_output, test_output):
    if one_prediction == real_output:
        status = "correct"
    else:
        status = "wrong"
    print(status, "| predicted=", one_prediction, "actual=", real_output, "features=", one_input)
