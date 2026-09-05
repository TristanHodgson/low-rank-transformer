import os
import sys

import boto3


local_path = sys.argv[1]
object_key = sys.argv[2]

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)

s3.upload_file(
    local_path,
    os.environ["R2_BUCKET"],
    object_key,
)

print(
    f"Uploaded {local_path} to "
    f"r2://{os.environ['R2_BUCKET']}/{object_key}"
)