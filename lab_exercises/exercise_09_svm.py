"""Exercise 9: Support Vector Machine (SVM) classification."""

# Step 1: Import required tools.
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Step 2: Load dataset.
cancer_data = load_breast_cancer()

# Step 3: Split data into training and test sets.
train_input, test_input, train_output, test_output = train_test_split(
    cancer_data.data, cancer_data.target, test_size=0.25, random_state=42
)

# Step 4: Create scaler to normalize feature ranges.
scaler = StandardScaler()

# Step 5: Fit scaler on training data and transform it.
train_input_scaled = scaler.fit_transform(train_input)

# Step 6: Transform test data using same scaler.
test_input_scaled = scaler.transform(test_input)

# Step 7: Create SVM model and train it.
model = SVC(kernel="rbf", gamma="scale", random_state=42)
model.fit(train_input_scaled, train_output)

# Step 8: Compute and print accuracy.
accuracy = model.score(test_input_scaled, test_output)
print("Test accuracy:", round(accuracy, 2))
