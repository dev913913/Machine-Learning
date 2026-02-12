"""Exercise 6: Bayesian Network for simplified heart-disease data."""

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork


def main() -> None:
    # Load the simplified heart-disease dataset.
    data = pd.read_csv("data/heart_disease_sample.csv")

    # Define the network structure: three inputs to one output.
    model = BayesianNetwork(
        [
            ("age_group", "heart_disease"),
            ("cholesterol", "heart_disease"),
            ("exercise_angina", "heart_disease"),
        ]
    )
    # Learn probabilities (CPDs) from the data.
    model.fit(data, estimator=MaximumLikelihoodEstimator)

    # Show the learned probabilities for each node.
    print("Learned CPDs:")
    for cpd in model.get_cpds():
        print(cpd)

    # Use inference to ask a medical question.
    inference = VariableElimination(model)
    result = inference.query(
        variables=["heart_disease"],
        evidence={"age_group": "senior", "cholesterol": "high", "exercise_angina": "yes"},
    )
    print("\nInference result for senior + high cholesterol + angina:")
    print(result)


if __name__ == "__main__":
    main()
