#!/bin/bash

cd /workspace/low-rank-transformer

pip install -r requirements.txt

# For the current test:
nvidia-smi > output.txt 2>&1

# Later replace the previous line with:
# python -u main.py > output.txt 2>&1

MAIN_EXIT=$?

cd /workspace

tar -czf low-rank-transformer.tar.gz low-rank-transformer
TAR_EXIT=$?

if [ "$TAR_EXIT" -eq 0 ]; then
    curl \
        --fail \
        --retry 5 \
        --retry-delay 5 \
        --retry-all-errors \
        --user "$NEXTCLOUD_USERNAME:$NEXTCLOUD_APP_PASSWORD" \
        -T low-rank-transformer.tar.gz \
        "$NEXTCLOUD_URL/remote.php/dav/files/$NEXTCLOUD_USERNAME/low-rank-transformer-$RUNPOD_POD_ID.tar.gz"
fi

cd /workspace/low-rank-transformer

python deploy/runpod-destroy.py

exit "$MAIN_EXIT"
