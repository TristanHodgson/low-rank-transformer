#!/bin/bash

cd /workspace/low-rank-transformer

pip install -r requirements.txt

python -u main.py > output.txt 2>&1
MAIN_EXIT=$?

cd /workspace

tar -czf low-rank-transformer.tar.gz low-rank-transformer
TAR_EXIT=$?

if [ "$TAR_EXIT" -eq 0 ]; then
    cd /workspace/low-rank-transformer

    python deploy/r2-upload.py \
        /workspace/low-rank-transformer.tar.gz \
        "runs/$RUNPOD_POD_ID/low-rank-transformer.tar.gz"

    UPLOAD_EXIT=$?
else
    UPLOAD_EXIT=1
fi

echo "Training exit code: $MAIN_EXIT"
echo "Archive exit code: $TAR_EXIT"
echo "Upload exit code: $UPLOAD_EXIT"

cd /workspace/low-rank-transformer

python deploy/runpod-destroy.py

exit "$MAIN_EXIT"