import os
from pathlib import Path

import runpod
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

runpod.api_key = os.environ["RUNPOD_API_KEY"]

pod = runpod.create_pod(
    name="low-rank-transformer",
    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    # gpu_type_id="NVIDIA RTX A5000",
    gpu_type_id="NVIDIA GeForce RTX 3090",
    cloud_type="COMMUNITY",
    gpu_count=1,
    container_disk_in_gb=50,
    volume_in_gb=0,

    docker_args=(
        "bash -lc '"
        "git clone "
        "https://github.com/TristanHodgson/low-rank-transformer.git "
        "/workspace/low-rank-transformer "
        "&& bash /workspace/low-rank-transformer/deploy/runpod-run.sh"
        "'"
    ),

    env={
        "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
        "R2_ACCESS_KEY_ID": os.environ["R2_ACCESS_KEY_ID"],
        "R2_SECRET_ACCESS_KEY": os.environ["R2_SECRET_ACCESS_KEY"],
        "R2_ENDPOINT": os.environ["R2_ENDPOINT"],
        "R2_BUCKET": os.environ["R2_BUCKET"],
    },
)

print(f"Started pod: {pod['id']}")
print("The job is now independent of this computer.")