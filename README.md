# Low Rank Approximation of Transformers for Decryption of Simple Ciphers

## To Run

### Locally

1. [Install PyTorch using the relevant command](https://pytorch.org/get-started/locally/)
2. Install other requirements: `pip install -r requirements.txt`

### RunPod

1. Set the `.env` file in the root of the repo as below

```
RUNPOD_API_KEY=

R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_ENDPOINT=
R2_BUCKET=
```

2. Install requirements: `pip install -r requirements.txt`
3. Run `python deploy/runpod-deploy.py`


## Results

| Rank Policy  |   Train Loss |   Train Char Acc |   Train Seq Acc |   Val Loss |   Val Char Acc |   Val Seq Acc |
|--------------|--------------|------------------|-----------------|------------|----------------|---------------|
| Full         |   0.00474626 |         0.998664 |        0.991962 |  0.0386182 |       0.990313 |      0.969311 |
| All rank 10           |   2.52928    |         0.337604 |        0        |  2.53258   |       0.337023 |      0        |

