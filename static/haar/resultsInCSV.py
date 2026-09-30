import csv
import random

# Define the table size
table_width = 10
table_height = 17

# Generate random values for the table
table = [[random.randint(1, 100) for _ in range(table_width)] for _ in range(table_height)]

# Define the CSV file path
csv_file = "table_data.csv"

# Write the table to the CSV file
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(table)

print("Table data saved to", csv_file)
