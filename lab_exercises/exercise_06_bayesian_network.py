"""Exercise 6: Bayesian Network with simple heart-disease sample data."""

# Step 1: Import required tools.
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

# Step 2: Read sample heart-disease data.
data_table = pd.read_csv("data/heart_disease_sample.csv")

# Step 3: Define network structure.
# Each pair (A, B) means A directly influences B.
model = DiscreteBayesianNetwork(
    [
        ("age_group", "heart_disease"),
        ("cholesterol", "heart_disease"),
        ("exercise_angina", "heart_disease"),
    ]
)

# Step 4: Learn probability tables from data.
model.fit(data_table, estimator=MaximumLikelihoodEstimator)

# Step 5: Print learned CPDs (conditional probability distributions).
print("Learned CPDs:")
for one_cpd in model.get_cpds():
    print(one_cpd)

# Step 6: Create inference tool for probability queries.
inference_tool = VariableElimination(model)

# Step 7: Ask probability of heart_disease for one evidence case.
result = inference_tool.query(
    variables=["heart_disease"],
    evidence={"age_group": "senior", "cholesterol": "high", "exercise_angina": "yes"},
)

# Step 8: Print the query result.
print("\nInference result for senior + high cholesterol + angina:")
print(result)
