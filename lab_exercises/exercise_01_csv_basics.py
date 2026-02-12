"""Exercise 1: Read CSV data and do very basic table operations."""

# Step 1: Import pandas.
# `import` is a Python keyword used to bring in a library.
# `as pd` gives pandas a short name so we can write less code.
import pandas as pd

# Step 2: Read the CSV file into a table.
# `=` stores the result on the right side into the variable on the left side.
data_table = pd.read_csv("data/numeric_data.csv")

# Step 3: Print the full table so beginners can see raw rows and columns.
print("Raw data:\n", data_table)

# Step 4: Print summary statistics for numeric columns.
# `describe()` gives common values like count, mean, min, max.
print("\nSummary statistics:\n", data_table.describe())

# Step 5: Print the mean (average) of each numeric column.
# `numeric_only=True` means: ignore non-numeric columns.
print("\nColumn means:\n", data_table.mean(numeric_only=True))

# Step 6: Print the sum (total) of each numeric column.
print("\nColumn sums:\n", data_table.sum(numeric_only=True))
