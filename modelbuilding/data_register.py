import os
from pathlib import Path
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing.")

repo_id = "chandrachurhghosh/predictive-maintenance-project"
repo_type = "dataset"

project_root = Path.cwd()
local_file = project_root / "data" / "raw" / "engine_data.csv"

print("Working directory:", project_root)
print("Looking for file:", local_file)
print("Local file exists:", local_file.exists())

if not local_file.exists():
    raise FileNotFoundError(f"File not found: {local_file}")

api = HfApi(token=HF_TOKEN)

create_repo(
    repo_id=repo_id,
    repo_type=repo_type,
    private=False,
    token=HF_TOKEN,
    exist_ok=True
)

df = pd.read_csv(local_file)
dataset = Dataset.from_pandas(df)

dataset.push_to_hub(
    repo_id,
    token=HF_TOKEN
)

print("Dataset uploaded successfully.")
