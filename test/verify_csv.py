import pandas as pd

# Load the generated CSV
generated_csv = pd.read_csv('results.csv')

# Load the expected CSV
expected_csv = pd.read_csv('./test/expected_results.csv')

# Check if the generated CSV matches the expected CSV
if generated_csv.equals(expected_csv):
    print("CSV output is correct. ✅")
    exit(0)
else:
    print("CSV output is incorrect. ❌")
    exit(1)