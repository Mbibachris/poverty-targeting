"""Upload the dashboard folder to its Hugging Face Space (run by CI on pushes to main).

Needs two environment variables: HF_SPACE (e.g. "user/space-name") and HF_TOKEN
(a Hugging Face token with write access, stored as a GitHub Actions secret).
"""

import os

from huggingface_hub import HfApi

space = os.environ["HF_SPACE"]
api = HfApi(token=os.environ["HF_TOKEN"])

# Creates the Space the first time; afterwards it already exists and this does nothing.
api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(
    repo_id=space,
    repo_type="space",
    folder_path="dashboard",
    ignore_patterns=["**/__pycache__/**", "__pycache__/**", "*.pyc"],
    commit_message=f"Deploy {os.environ.get('GITHUB_SHA', 'local')[:7]} from GitHub",
)
print(f"Deployed: https://huggingface.co/spaces/{space}")
