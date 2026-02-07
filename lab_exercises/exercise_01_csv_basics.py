"""Exercise 1: Read numeric CSV data and perform basic operations."""

import pandas as pd


def main() -> None:
    # Read the CSV file into a table (DataFrame).
    data = pd.read_csv("data/numeric_data.csv")

    # Print the full table.
    print("Raw data:\n", data)

    # Show common statistics like min, max, mean, and count.
    print("\nSummary statistics:\n", data.describe())

    # Compute the mean of numeric columns.
    print("\nColumn means:\n", data.mean(numeric_only=True))

    # Compute the sum of numeric columns.
    print("\nColumn sums:\n", data.sum(numeric_only=True))


if __name__ == "__main__":
    main()
