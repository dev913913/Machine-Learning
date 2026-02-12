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

# Step 4: Print first 5 rows only.
# `head()` shows rows from the top of the table.
print("\nFirst 5 rows (head):\n", data_table.head())

# Step 5: Print last 5 rows only.
# `tail()` shows rows from the bottom of the table.
print("\nLast 5 rows (tail):\n", data_table.tail())

# Step 6: Print first 3 rows.
# `head(3)` means "show only 3 rows from top".
print("\nFirst 3 rows (head(3)):\n", data_table.head(3))

# Step 7: Print random 2 rows.
# `sample(2)` picks 2 random rows from the table.
print("\nRandom 2 rows (sample(2)):\n", data_table.sample(2, random_state=42))

# Step 8: Print the number of rows and columns.
# `shape` returns a pair: (number_of_rows, number_of_columns).
print("\nTable shape (rows, columns):", data_table.shape)

# Step 9: Print column names.
# `columns` gives all column labels.
print("\nColumn names:", list(data_table.columns))

# Step 10: Print data type of each column.
# `dtypes` tells if a column has integers, text, etc.
print("\nColumn data types:\n", data_table.dtypes)

# Step 11: Print one column values.
# `data_table["value_a"]` means select only column named value_a.
print("\nValues of column value_a:\n", data_table["value_a"])

# Step 12: Add a new column.
# Here we create `value_a_plus_value_b` by adding value_a and value_b for each row.
data_table["value_a_plus_value_b"] = data_table["value_a"] + data_table["value_b"]
print("\nTable with new column value_a_plus_value_b:\n", data_table)

# Step 13: Filter rows using a condition.
# This keeps only rows where value_a is greater than 12.
rows_with_value_a_gt_12 = data_table[data_table["value_a"] > 12]
print("\nRows where value_a > 12:\n", rows_with_value_a_gt_12)

# Step 14: Check missing values.
# `isnull()` makes True/False table and `sum()` counts True values per column.
print("\nMissing value count in each column:\n", data_table.isnull().sum())

# Step 15: Print summary statistics for numeric columns.
# `describe()` gives common values like count, mean, min, max.
print("\nSummary statistics:\n", data_table.describe())

# Step 16: Print mean (average) of each numeric column.
# `numeric_only=True` means: ignore non-numeric columns.
print("\nColumn means:\n", data_table.mean(numeric_only=True))

# Step 17: Print sum (total) of each numeric column.
print("\nColumn sums:\n", data_table.sum(numeric_only=True))
