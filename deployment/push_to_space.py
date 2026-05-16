
import os
from huggingface_hub import HfApi, create_repo, upload_folder

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing.")

SPACE_REPO_ID = "chandrachurhghosh/predictive-maintenance-streamlit-app"
DEPLOYMENT_FOLDER = "deployment"

api = HfApi(token=HF_TOKEN)

create_repo(
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    space_sdk="docker",
    private=False,
    token=HF_TOKEN,
    exist_ok=True
)

upload_folder(
    folder_path=DEPLOYMENT_FOLDER,
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    token=HF_TOKEN
)

print(f"Deployment files pushed successfully to Hugging Face Space: {SPACE_REPO_ID}")
