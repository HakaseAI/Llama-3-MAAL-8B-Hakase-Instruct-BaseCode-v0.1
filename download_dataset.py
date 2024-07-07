from os import getenv

import huggingface_hub

huggingface_hub.login(
    token=getenv("HUGGINGFACE_TOKEN")
)
try:
    huggingface_hub.snapshot_download(repo_id="HakaseAI/HakaseDataset", local_dir="./dataset", repo_type="dataset")
except FileNotFoundError:
    pass
