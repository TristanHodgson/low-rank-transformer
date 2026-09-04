import os
import shlex
from pathlib import Path

import runpod
from dotenv import load_dotenv


# Load .env from the repository root.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

runpod.api_key = os.environ["RUNPOD_API_KEY"]

command = r"""
cd /workspace

git clone https://github.com/TristanHodgson/low-rank-transformer.git
cd low-rank-transformer

pip install -r requirements.txt

# Run training and save both stdout and stderr.
python -u main.py > output.txt 2>&1
MAIN_EXIT=$?

# Archive the entire repository, including the model, images and output.txt.
cd /workspace
tar -czf low-rank-transformer.tar.gz low-rank-transformer
TAR_EXIT=$?

# Upload to Nextcloud. Retry transient network failures.
if [ $TAR_EXIT -eq 0 ]; then
    curl --fail \
            --retry 5 \
            --retry-delay 5 \
            --retry-all-errors \
            --user "$NEXTCLOUD_USERNAME:$NEXTCLOUD_APP_PASSWORD" \
            -T low-rank-transformer.tar.gz \
            "$NEXTCLOUD_URL/remote.php/dav/files/$NEXTCLOUD_USERNAME/low-rank-transformer-$RUNPOD_POD_ID.tar.gz"
    UPLOAD_EXIT=$?
else
    UPLOAD_EXIT=1
fi

# Destroy the pod whether training or upload succeeded or failed.
cd /workspace/low-rank-transformer
python deploy/runpod-destroy.py

exit $MAIN_EXIT
"""

pod = runpod.create_pod(
    name="low-rank-transformer",
    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpu_type_id="NVIDIA RTX A5000",
    cloud_type="COMMUNITY",
    gpu_count=1,

    # Ephemeral storage only.
    container_disk_in_gb=50,
    volume_in_gb=0,

    docker_args=f"bash -lc {shlex.quote(command)}",

    env={
        "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
        "NEXTCLOUD_USERNAME": os.environ["NEXTCLOUD_USERNAME"],
        "NEXTCLOUD_APP_PASSWORD": os.environ["NEXTCLOUD_APP_PASSWORD"],
        "NEXTCLOUD_URL": os.environ["NEXTCLOUD_URL"],
    },
)

print(f"Started pod: {pod['id']}")
print("The job is now independent of this computer.")
