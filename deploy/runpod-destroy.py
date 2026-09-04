import os

import runpod


runpod.api_key = os.environ["RUNPOD_API_KEY"]

pod_id = os.environ["RUNPOD_POD_ID"]

print(f"Terminating pod {pod_id}...")
