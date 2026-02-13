import pandas as pd
from datasets import load_dataset

print("Downloading real Kaggle phishing dataset from Hugging Face...")

# Pulls a real-world cybersecurity dataset
dataset = load_dataset("shawhin/phishing-site-classification", split="train")

# Convert to a Pandas DataFrame
df = dataset.to_pandas()

# Rename the 'text' column to 'url' so it perfectly matches your engine
df = df.rename(columns={'text': 'url'})

# Save it to your folder
df.to_csv("url_dataset.csv", index=False)
print(f"Success! Downloaded {len(df)} real-world cyber threat URLs to url_dataset.csv.")