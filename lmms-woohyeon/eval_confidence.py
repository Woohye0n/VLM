import pandas as pd
import ast

# Load CSV data
df = pd.read_csv('matched_output.csv')
print(df.head())

# Convert 'Cumulative Confidences' from string representation of lists to actual lists
df['Cumulative Confidences'] = df['Cumulative Confidences'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Mark hallucination based on the Score (1.0 = no hallucination, <1.0 = hallucination)
df['Hallucination'] = df['Score'] < 1.0

# Calculate average confidence for each row by taking the mean of Cumulative Confidences
df['Average Confidence'] = df['Cumulative Confidences'].apply(lambda confs: sum(confs) / len(confs) if confs else None)

# Separate hallucinated and non-hallucinated rows and calculate the mean confidence for each
hallucinated_avg_conf = df[df['Hallucination']]['Average Confidence'].mean()
non_hallucinated_avg_conf = df[~df['Hallucination']]['Average Confidence'].mean()

print(f"Average confidence when hallucination exists: {hallucinated_avg_conf}")
print(f"Average confidence when hallucination does not exist: {non_hallucinated_avg_conf}")

import pandas as pd
import ast
import matplotlib.pyplot as plt

# Load CSV data
df = pd.read_csv('matched_output.csv')  # Update to the correct file path

# Convert 'CumulativeConfidences' from string representation of lists to actual lists
df['Cumulative Confidences'] = df['Cumulative Confidences'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Filter to include only rows for "Stage 2"
df_stage2 = df[df['Stage'] == 'Stage 2']

# Mark hallucination based on the Score (1.0 = no hallucination, <1.0 = hallucination)
df_stage2['No Hallucination'] = df_stage2['Score'] == 1.0

# Extract cumulative confidences for hallucinated cases and flatten them into a single list
hallucinated_confidences = [conf for confs in df_stage2[df_stage2['No Hallucination']]['Cumulative Confidences'] for conf in confs]

# Define bins for confidence ranges (e.g., 0.5–0.6, 0.6–0.7, etc.)
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bin_labels = [f"{bins[i]}~{bins[i+1]}" for i in range(len(bins)-1)]

# Create a histogram of the confidence values for hallucinated cases
hist = pd.cut(hallucinated_confidences, bins=bins, labels=bin_labels, right=False, include_lowest=True).value_counts()

# Plotting the histogram as a bar chart and saving it to a file
plt.figure(figsize=(8, 6))
hist.sort_index().plot(kind="bar")
plt.xlabel("Confidence Range")
plt.ylabel("Number of Cases")
plt.title("Distribution of Cumulative Confidence Levels When Hallucination Doe not Exists (Stage 2 Only)")
plt.savefig("confidence_distribution_stage2_nohal.png")  # Save the plot as a PNG file

import pandas as pd
import ast
import re

# Load CSV data
df = pd.read_csv('matched_output.csv')

# Convert 'CumulativeConfidences' from string representation of lists to actual lists
df['Cumulative Confidences'] = df['Cumulative Confidences'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Split cumulative confidences into separate columns
df[['Cumulative Confidence 1', 'Cumulative Confidence 2']] = pd.DataFrame(df['Cumulative Confidences'].tolist(), index=df.index)

# Normalize 'TextOutput' by removing non-alphanumeric characters for consistent comparison
df['NormalizedTextOutput'] = df['Text Output'].apply(lambda x: re.sub(r'\W+', '', str(x)).lower())

# Ensure unique rows for each DocID and Stage combination by aggregating or keeping the first occurrence
df = df.drop_duplicates(subset=['Doc ID', 'Stage'], keep='first')

# Pivot the DataFrame to compare normalized 'TextOutput' values between Stage 1 and Stage 2 for each DocID
pivot_df = df.pivot_table(index='Doc ID', columns='Stage', values='NormalizedTextOutput', aggfunc='first').reset_index()

# Create the 'Is Changed' column based on whether 'NormalizedTextOutput' differs between Stage 1 and Stage 2
pivot_df['Is Changed'] = pivot_df['Stage 1'] != pivot_df['Stage 2']

# Merge the 'Is Changed' column back to the original DataFrame
df = df.merge(pivot_df[['Doc ID', 'Is Changed']], on='Doc ID', how='left')

# Select and reorder the columns for the final output, excluding the 'NormalizedTextOutput'
df = df[['Doc ID', 'Question', 'Answer', 'Ground Truth', 'Prediction', 'Score', 'Stage', 
         'Cumulative Confidence 1', 'Cumulative Confidence 2', 'Is Changed']]

# Save the result to a new CSV file
df.to_csv('processed_output.csv', index=False)





