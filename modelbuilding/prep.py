
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing.")

api = HfApi(token=HF_TOKEN)

REPO_ID = "chandrachurhghosh/predictive-maintenance-project"
REPO_TYPE = "dataset"

DATASET_PATH = "hf://datasets/chandrachurhghosh/predictive-maintenance-project/data/raw/engine_data.csv"

print("Loading dataset from Hugging Face...")
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# No unnecessary columns to remove

# Target column
target_col = "Engine_Condition"

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}")

# Target is already an integer column

# Ensure numeric columns are properly converted
for col in df.columns:
    if col != target_col:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# No missing values to be treated / imputed

# Define features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save processed files locally
processed_path = "/content/predictive-maintenance-project/data/processed"
os.makedirs(processed_path, exist_ok=True)

X_train.to_csv(f"{processed_path}/X_train.csv", index=False)
X_test.to_csv(f"{processed_path}/X_test.csv", index=False)
y_train.to_csv(f"{processed_path}/y_train.csv", index=False)
y_test.to_csv(f"{processed_path}/y_test.csv", index=False)

print("Train-test files saved locally.")

# Upload processed files to Hugging Face
files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

for file in files:
    api.upload_file(
        path_or_fileobj=f"{processed_path}/{file}",
        path_in_repo=f"data/processed/{file}",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=HF_TOKEN
    )
    print(f"Uploaded {file} to Hugging Face.")

print("Data preparation completed successfully.")
