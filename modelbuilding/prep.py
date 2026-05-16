
import os
from pathlib import Path
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
print("Original Columns:", df.columns.tolist())

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

print("Cleaned Columns:", df.columns.tolist())

target_col = "Engine_Condition"

if target_col not in df.columns:
    raise ValueError(
        f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}"
    )

# Convert all columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Validate missing values
missing_counts = df.isnull().sum()
print("Missing values after numeric conversion:")
print(missing_counts)

# If any missing values appear after conversion, fill with median
for col in df.columns:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        print(f"Filled missing values in {col} with median: {median_value}")

# Ensure target is integer
df[target_col] = df[target_col].astype(int)

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Prepare paths
project_root = Path.cwd()
processed_path = project_root / "data" / "processed"
processed_path.mkdir(parents=True, exist_ok=True)

X_train.to_csv(processed_path / "X_train.csv", index=False)
X_test.to_csv(processed_path / "X_test.csv", index=False)
y_train.to_csv(processed_path / "y_train.csv", index=False)
y_test.to_csv(processed_path / "y_test.csv", index=False)

print("Train-test files saved locally at:", processed_path)

files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

for file in files:
    api.upload_file(
        path_or_fileobj=str(processed_path / file),
        path_in_repo=f"data/processed/{file}",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=HF_TOKEN
    )
    print(f"Uploaded {file} to Hugging Face.")

print("Data preparation completed successfully.")
