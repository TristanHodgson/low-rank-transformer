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

### Policies

#### Ranks of Matrices in R10

| Layer                         | Rank | Original Rank | Module Name           |
| ----------------------------- | ---- | ------------- | --------------------- |
| Block 0.Sa.Q Projection       | 10   | 768           | blocks.0.sa.q_proj    |
| Block 0.Sa.K Projection       | 10   | 768           | blocks.0.sa.k_proj    |
| Block 0.Sa.V Projection       | 10   | 768           | blocks.0.sa.v_proj    |
| Block 0.Sa.Output Projection  | 10   | 768           | blocks.0.sa.out_proj  |
| Block 0.Ffwd.Net.0            | 10   | 768           | blocks.0.ffwd.net.0   |
| Block 0.Ffwd.Net.2            | 10   | 768           | blocks.0.ffwd.net.2   |
| Block 1.Sa.Q Projection       | 10   | 768           | blocks.1.sa.q_proj    |
| Block 1.Sa.K Projection       | 10   | 768           | blocks.1.sa.k_proj    |
| Block 1.Sa.V Projection       | 10   | 768           | blocks.1.sa.v_proj    |
| Block 1.Sa.Output Projection  | 10   | 768           | blocks.1.sa.out_proj  |
| Block 1.Ffwd.Net.0            | 10   | 768           | blocks.1.ffwd.net.0   |
| Block 1.Ffwd.Net.2            | 10   | 768           | blocks.1.ffwd.net.2   |
| Block 2.Sa.Q Projection       | 10   | 768           | blocks.2.sa.q_proj    |
| Block 2.Sa.K Projection       | 10   | 768           | blocks.2.sa.k_proj    |
| Block 2.Sa.V Projection       | 10   | 768           | blocks.2.sa.v_proj    |
| Block 2.Sa.Output Projection  | 10   | 768           | blocks.2.sa.out_proj  |
| Block 2.Ffwd.Net.0            | 10   | 768           | blocks.2.ffwd.net.0   |
| Block 2.Ffwd.Net.2            | 10   | 768           | blocks.2.ffwd.net.2   |
| Block 3.Sa.Q Projection       | 10   | 768           | blocks.3.sa.q_proj    |
| Block 3.Sa.K Projection       | 10   | 768           | blocks.3.sa.k_proj    |
| Block 3.Sa.V Projection       | 10   | 768           | blocks.3.sa.v_proj    |
| Block 3.Sa.Output Projection  | 10   | 768           | blocks.3.sa.out_proj  |
| Block 3.Ffwd.Net.0            | 10   | 768           | blocks.3.ffwd.net.0   |
| Block 3.Ffwd.Net.2            | 10   | 768           | blocks.3.ffwd.net.2   |
| Block 4.Sa.Q Projection       | 10   | 768           | blocks.4.sa.q_proj    |
| Block 4.Sa.K Projection       | 10   | 768           | blocks.4.sa.k_proj    |
| Block 4.Sa.V Projection       | 10   | 768           | blocks.4.sa.v_proj    |
| Block 4.Sa.Output Projection  | 10   | 768           | blocks.4.sa.out_proj  |
| Block 4.Ffwd.Net.0            | 10   | 768           | blocks.4.ffwd.net.0   |
| Block 4.Ffwd.Net.2            | 10   | 768           | blocks.4.ffwd.net.2   |
| Block 5.Sa.Q Projection       | 10   | 768           | blocks.5.sa.q_proj    |
| Block 5.Sa.K Projection       | 10   | 768           | blocks.5.sa.k_proj    |
| Block 5.Sa.V Projection       | 10   | 768           | blocks.5.sa.v_proj    |
| Block 5.Sa.Output Projection  | 10   | 768           | blocks.5.sa.out_proj  |
| Block 5.Ffwd.Net.0            | 10   | 768           | blocks.5.ffwd.net.0   |
| Block 5.Ffwd.Net.2            | 10   | 768           | blocks.5.ffwd.net.2   |
| Block 6.Sa.Q Projection       | 10   | 768           | blocks.6.sa.q_proj    |
| Block 6.Sa.K Projection       | 10   | 768           | blocks.6.sa.k_proj    |
| Block 6.Sa.V Projection       | 10   | 768           | blocks.6.sa.v_proj    |
| Block 6.Sa.Output Projection  | 10   | 768           | blocks.6.sa.out_proj  |
| Block 6.Ffwd.Net.0            | 10   | 768           | blocks.6.ffwd.net.0   |
| Block 6.Ffwd.Net.2            | 10   | 768           | blocks.6.ffwd.net.2   |
| Block 7.Sa.Q Projection       | 10   | 768           | blocks.7.sa.q_proj    |
| Block 7.Sa.K Projection       | 10   | 768           | blocks.7.sa.k_proj    |
| Block 7.Sa.V Projection       | 10   | 768           | blocks.7.sa.v_proj    |
| Block 7.Sa.Output Projection  | 10   | 768           | blocks.7.sa.out_proj  |
| Block 7.Ffwd.Net.0            | 10   | 768           | blocks.7.ffwd.net.0   |
| Block 7.Ffwd.Net.2            | 10   | 768           | blocks.7.ffwd.net.2   |
| Block 8.Sa.Q Projection       | 10   | 768           | blocks.8.sa.q_proj    |
| Block 8.Sa.K Projection       | 10   | 768           | blocks.8.sa.k_proj    |
| Block 8.Sa.V Projection       | 10   | 768           | blocks.8.sa.v_proj    |
| Block 8.Sa.Output Projection  | 10   | 768           | blocks.8.sa.out_proj  |
| Block 8.Ffwd.Net.0            | 10   | 768           | blocks.8.ffwd.net.0   |
| Block 8.Ffwd.Net.2            | 10   | 768           | blocks.8.ffwd.net.2   |
| Block 9.Sa.Q Projection       | 10   | 768           | blocks.9.sa.q_proj    |
| Block 9.Sa.K Projection       | 10   | 768           | blocks.9.sa.k_proj    |
| Block 9.Sa.V Projection       | 10   | 768           | blocks.9.sa.v_proj    |
| Block 9.Sa.Output Projection  | 10   | 768           | blocks.9.sa.out_proj  |
| Block 9.Ffwd.Net.0            | 10   | 768           | blocks.9.ffwd.net.0   |
| Block 9.Ffwd.Net.2            | 10   | 768           | blocks.9.ffwd.net.2   |
| Block 10.Sa.Q Projection      | 10   | 768           | blocks.10.sa.q_proj   |
| Block 10.Sa.K Projection      | 10   | 768           | blocks.10.sa.k_proj   |
| Block 10.Sa.V Projection      | 10   | 768           | blocks.10.sa.v_proj   |
| Block 10.Sa.Output Projection | 10   | 768           | blocks.10.sa.out_proj |
| Block 10.Ffwd.Net.0           | 10   | 768           | blocks.10.ffwd.net.0  |
| Block 10.Ffwd.Net.2           | 10   | 768           | blocks.10.ffwd.net.2  |
| Block 11.Sa.Q Projection      | 10   | 768           | blocks.11.sa.q_proj   |
| Block 11.Sa.K Projection      | 10   | 768           | blocks.11.sa.k_proj   |
| Block 11.Sa.V Projection      | 10   | 768           | blocks.11.sa.v_proj   |
| Block 11.Sa.Output Projection | 10   | 768           | blocks.11.sa.out_proj |
| Block 11.Ffwd.Net.0           | 10   | 768           | blocks.11.ffwd.net.0  |
| Block 11.Ffwd.Net.2           | 10   | 768           | blocks.11.ffwd.net.2  |

#### Ranks of Matrices in Energy95

| Layer                         | Rank | Original Rank | Module Name           |
| ----------------------------- | ---- | ------------- | --------------------- |
| Block 0.Sa.Q Projection       | 589  | 768           | blocks.0.sa.q_proj    |
| Block 0.Sa.K Projection       | 589  | 768           | blocks.0.sa.k_proj    |
| Block 0.Sa.V Projection       | 590  | 768           | blocks.0.sa.v_proj    |
| Block 0.Sa.Output Projection  | 589  | 768           | blocks.0.sa.out_proj  |
| Block 0.Ffwd.Net.0            | 703  | 768           | blocks.0.ffwd.net.0   |
| Block 0.Ffwd.Net.2            | 703  | 768           | blocks.0.ffwd.net.2   |
| Block 1.Sa.Q Projection       | 589  | 768           | blocks.1.sa.q_proj    |
| Block 1.Sa.K Projection       | 589  | 768           | blocks.1.sa.k_proj    |
| Block 1.Sa.V Projection       | 590  | 768           | blocks.1.sa.v_proj    |
| Block 1.Sa.Output Projection  | 590  | 768           | blocks.1.sa.out_proj  |
| Block 1.Ffwd.Net.0            | 703  | 768           | blocks.1.ffwd.net.0   |
| Block 1.Ffwd.Net.2            | 703  | 768           | blocks.1.ffwd.net.2   |
| Block 2.Sa.Q Projection       | 589  | 768           | blocks.2.sa.q_proj    |
| Block 2.Sa.K Projection       | 589  | 768           | blocks.2.sa.k_proj    |
| Block 2.Sa.V Projection       | 590  | 768           | blocks.2.sa.v_proj    |
| Block 2.Sa.Output Projection  | 590  | 768           | blocks.2.sa.out_proj  |
| Block 2.Ffwd.Net.0            | 703  | 768           | blocks.2.ffwd.net.0   |
| Block 2.Ffwd.Net.2            | 703  | 768           | blocks.2.ffwd.net.2   |
| Block 3.Sa.Q Projection       | 589  | 768           | blocks.3.sa.q_proj    |
| Block 3.Sa.K Projection       | 589  | 768           | blocks.3.sa.k_proj    |
| Block 3.Sa.V Projection       | 590  | 768           | blocks.3.sa.v_proj    |
| Block 3.Sa.Output Projection  | 589  | 768           | blocks.3.sa.out_proj  |
| Block 3.Ffwd.Net.0            | 703  | 768           | blocks.3.ffwd.net.0   |
| Block 3.Ffwd.Net.2            | 703  | 768           | blocks.3.ffwd.net.2   |
| Block 4.Sa.Q Projection       | 588  | 768           | blocks.4.sa.q_proj    |
| Block 4.Sa.K Projection       | 589  | 768           | blocks.4.sa.k_proj    |
| Block 4.Sa.V Projection       | 590  | 768           | blocks.4.sa.v_proj    |
| Block 4.Sa.Output Projection  | 589  | 768           | blocks.4.sa.out_proj  |
| Block 4.Ffwd.Net.0            | 703  | 768           | blocks.4.ffwd.net.0   |
| Block 4.Ffwd.Net.2            | 703  | 768           | blocks.4.ffwd.net.2   |
| Block 5.Sa.Q Projection       | 589  | 768           | blocks.5.sa.q_proj    |
| Block 5.Sa.K Projection       | 589  | 768           | blocks.5.sa.k_proj    |
| Block 5.Sa.V Projection       | 590  | 768           | blocks.5.sa.v_proj    |
| Block 5.Sa.Output Projection  | 590  | 768           | blocks.5.sa.out_proj  |
| Block 5.Ffwd.Net.0            | 703  | 768           | blocks.5.ffwd.net.0   |
| Block 5.Ffwd.Net.2            | 703  | 768           | blocks.5.ffwd.net.2   |
| Block 6.Sa.Q Projection       | 589  | 768           | blocks.6.sa.q_proj    |
| Block 6.Sa.K Projection       | 589  | 768           | blocks.6.sa.k_proj    |
| Block 6.Sa.V Projection       | 589  | 768           | blocks.6.sa.v_proj    |
| Block 6.Sa.Output Projection  | 590  | 768           | blocks.6.sa.out_proj  |
| Block 6.Ffwd.Net.0            | 703  | 768           | blocks.6.ffwd.net.0   |
| Block 6.Ffwd.Net.2            | 703  | 768           | blocks.6.ffwd.net.2   |
| Block 7.Sa.Q Projection       | 589  | 768           | blocks.7.sa.q_proj    |
| Block 7.Sa.K Projection       | 589  | 768           | blocks.7.sa.k_proj    |
| Block 7.Sa.V Projection       | 590  | 768           | blocks.7.sa.v_proj    |
| Block 7.Sa.Output Projection  | 590  | 768           | blocks.7.sa.out_proj  |
| Block 7.Ffwd.Net.0            | 703  | 768           | blocks.7.ffwd.net.0   |
| Block 7.Ffwd.Net.2            | 703  | 768           | blocks.7.ffwd.net.2   |
| Block 8.Sa.Q Projection       | 589  | 768           | blocks.8.sa.q_proj    |
| Block 8.Sa.K Projection       | 589  | 768           | blocks.8.sa.k_proj    |
| Block 8.Sa.V Projection       | 590  | 768           | blocks.8.sa.v_proj    |
| Block 8.Sa.Output Projection  | 590  | 768           | blocks.8.sa.out_proj  |
| Block 8.Ffwd.Net.0            | 703  | 768           | blocks.8.ffwd.net.0   |
| Block 8.Ffwd.Net.2            | 703  | 768           | blocks.8.ffwd.net.2   |
| Block 9.Sa.Q Projection       | 590  | 768           | blocks.9.sa.q_proj    |
| Block 9.Sa.K Projection       | 590  | 768           | blocks.9.sa.k_proj    |
| Block 9.Sa.V Projection       | 590  | 768           | blocks.9.sa.v_proj    |
| Block 9.Sa.Output Projection  | 590  | 768           | blocks.9.sa.out_proj  |
| Block 9.Ffwd.Net.0            | 703  | 768           | blocks.9.ffwd.net.0   |
| Block 9.Ffwd.Net.2            | 703  | 768           | blocks.9.ffwd.net.2   |
| Block 10.Sa.Q Projection      | 590  | 768           | blocks.10.sa.q_proj   |
| Block 10.Sa.K Projection      | 590  | 768           | blocks.10.sa.k_proj   |
| Block 10.Sa.V Projection      | 589  | 768           | blocks.10.sa.v_proj   |
| Block 10.Sa.Output Projection | 590  | 768           | blocks.10.sa.out_proj |
| Block 10.Ffwd.Net.0           | 703  | 768           | blocks.10.ffwd.net.0  |
| Block 10.Ffwd.Net.2           | 703  | 768           | blocks.10.ffwd.net.2  |
| Block 11.Sa.Q Projection      | 589  | 768           | blocks.11.sa.q_proj   |
| Block 11.Sa.K Projection      | 590  | 768           | blocks.11.sa.k_proj   |
| Block 11.Sa.V Projection      | 590  | 768           | blocks.11.sa.v_proj   |
| Block 11.Sa.Output Projection | 590  | 768           | blocks.11.sa.out_proj |
| Block 11.Ffwd.Net.0           | 703  | 768           | blocks.11.ffwd.net.0  |
| Block 11.Ffwd.Net.2           | 703  | 768           | blocks.11.ffwd.net.2  |

#### Ranks of Matrices in Q10

| Layer                         | Rank | Original Rank | Module Name           |
| ----------------------------- | ---- | ------------- | --------------------- |
| Block 0.Sa.Q Projection       | 10   | 768           | blocks.0.sa.q_proj    |
| Block 0.Sa.K Projection       | 768  | 768           | blocks.0.sa.k_proj    |
| Block 0.Sa.V Projection       | 768  | 768           | blocks.0.sa.v_proj    |
| Block 0.Sa.Output Projection  | 768  | 768           | blocks.0.sa.out_proj  |
| Block 0.Ffwd.Net.0            | 768  | 768           | blocks.0.ffwd.net.0   |
| Block 0.Ffwd.Net.2            | 768  | 768           | blocks.0.ffwd.net.2   |
| Block 1.Sa.Q Projection       | 10   | 768           | blocks.1.sa.q_proj    |
| Block 1.Sa.K Projection       | 768  | 768           | blocks.1.sa.k_proj    |
| Block 1.Sa.V Projection       | 768  | 768           | blocks.1.sa.v_proj    |
| Block 1.Sa.Output Projection  | 768  | 768           | blocks.1.sa.out_proj  |
| Block 1.Ffwd.Net.0            | 768  | 768           | blocks.1.ffwd.net.0   |
| Block 1.Ffwd.Net.2            | 768  | 768           | blocks.1.ffwd.net.2   |
| Block 2.Sa.Q Projection       | 10   | 768           | blocks.2.sa.q_proj    |
| Block 2.Sa.K Projection       | 768  | 768           | blocks.2.sa.k_proj    |
| Block 2.Sa.V Projection       | 768  | 768           | blocks.2.sa.v_proj    |
| Block 2.Sa.Output Projection  | 768  | 768           | blocks.2.sa.out_proj  |
| Block 2.Ffwd.Net.0            | 768  | 768           | blocks.2.ffwd.net.0   |
| Block 2.Ffwd.Net.2            | 768  | 768           | blocks.2.ffwd.net.2   |
| Block 3.Sa.Q Projection       | 10   | 768           | blocks.3.sa.q_proj    |
| Block 3.Sa.K Projection       | 768  | 768           | blocks.3.sa.k_proj    |
| Block 3.Sa.V Projection       | 768  | 768           | blocks.3.sa.v_proj    |
| Block 3.Sa.Output Projection  | 768  | 768           | blocks.3.sa.out_proj  |
| Block 3.Ffwd.Net.0            | 768  | 768           | blocks.3.ffwd.net.0   |
| Block 3.Ffwd.Net.2            | 768  | 768           | blocks.3.ffwd.net.2   |
| Block 4.Sa.Q Projection       | 10   | 768           | blocks.4.sa.q_proj    |
| Block 4.Sa.K Projection       | 768  | 768           | blocks.4.sa.k_proj    |
| Block 4.Sa.V Projection       | 768  | 768           | blocks.4.sa.v_proj    |
| Block 4.Sa.Output Projection  | 768  | 768           | blocks.4.sa.out_proj  |
| Block 4.Ffwd.Net.0            | 768  | 768           | blocks.4.ffwd.net.0   |
| Block 4.Ffwd.Net.2            | 768  | 768           | blocks.4.ffwd.net.2   |
| Block 5.Sa.Q Projection       | 10   | 768           | blocks.5.sa.q_proj    |
| Block 5.Sa.K Projection       | 768  | 768           | blocks.5.sa.k_proj    |
| Block 5.Sa.V Projection       | 768  | 768           | blocks.5.sa.v_proj    |
| Block 5.Sa.Output Projection  | 768  | 768           | blocks.5.sa.out_proj  |
| Block 5.Ffwd.Net.0            | 768  | 768           | blocks.5.ffwd.net.0   |
| Block 5.Ffwd.Net.2            | 768  | 768           | blocks.5.ffwd.net.2   |
| Block 6.Sa.Q Projection       | 10   | 768           | blocks.6.sa.q_proj    |
| Block 6.Sa.K Projection       | 768  | 768           | blocks.6.sa.k_proj    |
| Block 6.Sa.V Projection       | 768  | 768           | blocks.6.sa.v_proj    |
| Block 6.Sa.Output Projection  | 768  | 768           | blocks.6.sa.out_proj  |
| Block 6.Ffwd.Net.0            | 768  | 768           | blocks.6.ffwd.net.0   |
| Block 6.Ffwd.Net.2            | 768  | 768           | blocks.6.ffwd.net.2   |
| Block 7.Sa.Q Projection       | 10   | 768           | blocks.7.sa.q_proj    |
| Block 7.Sa.K Projection       | 768  | 768           | blocks.7.sa.k_proj    |
| Block 7.Sa.V Projection       | 768  | 768           | blocks.7.sa.v_proj    |
| Block 7.Sa.Output Projection  | 768  | 768           | blocks.7.sa.out_proj  |
| Block 7.Ffwd.Net.0            | 768  | 768           | blocks.7.ffwd.net.0   |
| Block 7.Ffwd.Net.2            | 768  | 768           | blocks.7.ffwd.net.2   |
| Block 8.Sa.Q Projection       | 10   | 768           | blocks.8.sa.q_proj    |
| Block 8.Sa.K Projection       | 768  | 768           | blocks.8.sa.k_proj    |
| Block 8.Sa.V Projection       | 768  | 768           | blocks.8.sa.v_proj    |
| Block 8.Sa.Output Projection  | 768  | 768           | blocks.8.sa.out_proj  |
| Block 8.Ffwd.Net.0            | 768  | 768           | blocks.8.ffwd.net.0   |
| Block 8.Ffwd.Net.2            | 768  | 768           | blocks.8.ffwd.net.2   |
| Block 9.Sa.Q Projection       | 10   | 768           | blocks.9.sa.q_proj    |
| Block 9.Sa.K Projection       | 768  | 768           | blocks.9.sa.k_proj    |
| Block 9.Sa.V Projection       | 768  | 768           | blocks.9.sa.v_proj    |
| Block 9.Sa.Output Projection  | 768  | 768           | blocks.9.sa.out_proj  |
| Block 9.Ffwd.Net.0            | 768  | 768           | blocks.9.ffwd.net.0   |
| Block 9.Ffwd.Net.2            | 768  | 768           | blocks.9.ffwd.net.2   |
| Block 10.Sa.Q Projection      | 10   | 768           | blocks.10.sa.q_proj   |
| Block 10.Sa.K Projection      | 768  | 768           | blocks.10.sa.k_proj   |
| Block 10.Sa.V Projection      | 768  | 768           | blocks.10.sa.v_proj   |
| Block 10.Sa.Output Projection | 768  | 768           | blocks.10.sa.out_proj |
| Block 10.Ffwd.Net.0           | 768  | 768           | blocks.10.ffwd.net.0  |
| Block 10.Ffwd.Net.2           | 768  | 768           | blocks.10.ffwd.net.2  |
| Block 11.Sa.Q Projection      | 10   | 768           | blocks.11.sa.q_proj   |
| Block 11.Sa.K Projection      | 768  | 768           | blocks.11.sa.k_proj   |
| Block 11.Sa.V Projection      | 768  | 768           | blocks.11.sa.v_proj   |
| Block 11.Sa.Output Projection | 768  | 768           | blocks.11.sa.out_proj |
| Block 11.Ffwd.Net.0           | 768  | 768           | blocks.11.ffwd.net.0  |
| Block 11.Ffwd.Net.2           | 768  | 768           | blocks.11.ffwd.net.2  |

### Evaluating Policies

| Strategy | Train Loss  | Train Char Acc | Train Seq Acc | Val Loss  | Val Char Acc | Val Seq Acc |
| -------- | ----------- | -------------- | ------------- | --------- | ------------ | ----------- |
| Full     | 0.000150825 | 1              | 1             | 0.0280017 | 0.993109     | 0.98116     |
| R10      | 2.7005      | 0.281842       | 0             | 2.70294   | 0.281473     | 0           |
| Energy95 | 0.000181395 | 1              | 1             | 0.0275331 | 0.993063     | 0.980408    |
| Q10      | 0.0243967   | 0.992345       | 0.947816      | 0.0573864 | 0.983316     | 0.925426    |
