import numpy as np
import csv

csv_data = []

# Function to load CSV
def load_csv(file_path):
    global csv_data
    with open(file_path, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        csv_data = list(reader)

# Call this once at the start
load_csv('books.csv')

# Load matrix
doc_topic_matrix = np.load("sol.npy")

# Set formatting for cleaner output
np.set_printoptions(precision=4, suppress=True)

print("Full Document-Topic Matrix (Books × Topics):\n")

# Optional: skip header if present
start_index = 1 if csv_data[0][0].lower() == "title" else 0

# Print each book's topic distribution
for i, row in enumerate(doc_topic_matrix):
    book_title = csv_data[i + start_index][0]  # +1 if skipping header
    topic_weights = "  ".join(f"{weight:.4f}" for weight in row)
    print(f"{book_title[:50]:50} {i:>3}:  {topic_weights}")  # trims long titles
