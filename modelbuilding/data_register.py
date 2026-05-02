from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

repo_id = "chandrachurhghosh/predictive-maintenance-project"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Ensure dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating it...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

# Local file path (relative to repo root in GitHub Actions)
local_file = "/content/predictive-maintenance-project/data/raw/engine_data.csv"

# Target path in HF dataset
hf_path = "data/raw/engine_data.csv"

print("Working directory:", os.getcwd())
print("Local file exists:", os.path.isfile(local_file))

if not os.path.isfile(local_file):
    raise FileNotFoundError(f"File not found: {local_file}")

api.upload_file(
    path_or_fileobj=local_file,
    path_in_repo=hf_path,
    repo_id=repo_id,
    repo_type=repo_type
)

print(f"Uploaded {local_file} → hf://datasets/{repo_id}/{hf_path}")
