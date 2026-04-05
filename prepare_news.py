import pandas as pd

# Load CSVs
fake = pd.read_csv("Fake.csv", encoding='ISO-8859-1', low_memory=False)
true = pd.read_csv("True.csv", encoding='ISO-8859-1', low_memory=False)

# Keep only 'title' and 'text'
fake = fake[['title', 'text']]
true = true[['title', 'text']]

# Add a label column
fake['label'] = 'FAKE'
true['label'] = 'REAL'

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Optional: combine title + text into one column
data['text'] = data['title'] + " " + data['text']
data = data[['text', 'label']]  # final clean dataset

# Save cleaned dataset
data.to_csv("news_clean.csv", index=False)
print("Cleaned CSV created successfully as news_clean.csv!")
