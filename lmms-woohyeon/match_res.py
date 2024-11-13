import pandas as pd
import json

# Load JSON data from a file
with open('/home/aidas_intern_1/woohyeon/lmms-eval/logs/liuhaotian__llava-v1.6-vicuna-7b/20241106_153829_samples_pope.jsonl', 'r') as f:
    json_data = [json.loads(line) for line in f]

# Load CSV data
csv_df = pd.read_csv('/home/aidas_intern_1/woohyeon/lmms-eval/generation_output.csv')

# Flatten cumulative confidence columns for easier matching
csv_df["Cumulative Confidence"] = csv_df.apply(
    lambda row: [row[f"Cumulative Confidence {i+1}"] for i in range(10) if pd.notna(row.get(f"Cumulative Confidence {i+1}"))],
    axis=1
)

# Matching and extracting relevant fields
matched_data = []
for record in json_data:
    doc_id = record['doc_id']
    question = record['doc']['question']
    answer = record['doc']['answer']
    ground_truth = record['pope_accuracy']['ground_truth']
    prediction = record['pope_accuracy']['prediction']
    score = record['pope_accuracy']['score']
    
    # Filter CSV data for matching Doc ID
    stage_data = csv_df[csv_df["Doc ID"] == f"({doc_id},)"]

    # Separate stages for Stage 1 and Stage 2
    for stage, row in stage_data.iterrows():
        stage_name = row["Stage"]
        text_output = row["Text Output"]
        cumulative_confidences = row["Cumulative Confidence"]

        # Add to matched data list
        matched_data.append({
            "Doc ID": doc_id,
            "Question": question,
            "Answer": answer,
            "Ground Truth": ground_truth,
            "Prediction": prediction,
            "Score": score,
            "Stage": stage_name,
            "Text Output": text_output,
            "Cumulative Confidences": cumulative_confidences
        })

# Convert matched data to DataFrame and save to a new CSV
matched_df = pd.DataFrame(matched_data)
matched_df.to_csv('matched_output.csv', index=False)
