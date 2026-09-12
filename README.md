# Low Rank Approximation of Transformers for Decryption of Simple Ciphers

We train a BERT style transformer model trained to decrypt simple cesar ciphers. Once trained, the weight matrices are compressed using low-rank approximations via SVD. This enables us to benchmark how this method of model simplification impacts total parameter count, computational efficiency, and decryption accuracy.

## To Run

### Locally

1. [Install PyTorch using the relevant command](https://pytorch.org/get-started/locally/)
2. Install other requirements: `pip install -r requirements.txt`

### RunPod + Cloudflare R2

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

## Compression Strategies

| Strategy Name | Description                                                                                  |
| ------------- | -------------------------------------------------------------------------------------------- |
| R100          | All matrices compressed to rank 100, except for the final output layer                       |
| Energy95      | All matrices are compressed so that they have the top 95% of their singular values by weight |

## Results

| Strategy   |   Train Loss |   Train Char Acc |   Train Seq Acc |   Val Loss |   Val Char Acc |   Val Seq Acc |   Model Parameters Count |
|------------|--------------|------------------|-----------------|------------|----------------|---------------|--------------------------|
| Full       |   0.00332599 |         0.999201 |     0.99303     |  0.0409495 |       0.989817 |   0.968246    |                 85129760 |
| Energy0    |   3.18069    |         0.196566 |     0           |  3.18285   |       0.196095 |   0           |                   360992 |
| Energy5    |   2.11861    |         0.461214 |     0           |  2.12325   |       0.45987  |   0           |                  3754016 |
| Energy10   |   1.56346    |         0.62345  |     0.000596565 |  1.5709    |       0.620653 |   0.000595593 |                  7522592 |
| Energy15   |   1.12232    |         0.743981 |     0.0135012   |  1.13269   |       0.740971 |   0.0151719   |                 11419424 |
| Energy20   |   0.733543   |         0.840802 |     0.092656    |  0.747055  |       0.836379 |   0.0931005   |                 15518240 |
| Energy25   |   0.449714   |         0.903844 |     0.259663    |  0.466772  |       0.897538 |   0.256136    |                 19774496 |
| Energy30   |   0.259762   |         0.944557 |     0.478006    |  0.279951  |       0.937235 |   0.46986     |                 24138272 |
| Energy35   |   0.141369   |         0.969256 |     0.67999     |  0.164211  |       0.961526 |   0.666562    |                 28726304 |
| Energy40   |   0.0770271  |         0.982951 |     0.811423    |  0.102231  |       0.974632 |   0.795273    |                 33435680 |
| Energy45   |   0.0440232  |         0.989933 |     0.890232    |  0.0713392 |       0.981096 |   0.87154     |                 38383136 |
| Energy50   |   0.0275296  |         0.993564 |     0.928538    |  0.0562495 |       0.984376 |   0.90621     |                 43535648 |
| Energy55   |   0.0176528  |         0.995906 |     0.954567    |  0.0476842 |       0.986447 |   0.928748    |                 48897824 |
| Energy60   |   0.0119428  |         0.99719  |     0.970266    |  0.0428606 |       0.987762 |   0.944359    |                 54557216 |
| Energy65   |   0.00877924 |         0.997931 |     0.978461    |  0.0404494 |       0.988598 |   0.9516      |                 60492320 |
| Energy70   |   0.00662666 |         0.998441 |     0.984332    |  0.0389707 |       0.989116 |   0.957682    |                 66807584 |
| Energy75   |   0.00496924 |         0.998843 |     0.988791    |  0.0376105 |       0.989758 |   0.962509    |                 73559840 |
| Energy80   |   0.00430579 |         0.998976 |     0.990172    |  0.0376302 |       0.990029 |   0.964358    |                 80789024 |
| Energy85   |   0.00391541 |         0.999093 |     0.990957    |  0.038126  |       0.990039 |   0.965424    |                 88772384 |
| Energy90   |   0.00347743 |         0.999178 |     0.992339    |  0.0386118 |       0.990126 |   0.966772    |                 97729568 |
| Energy95   |   0.00342841 |         0.999166 |     0.992559    |  0.0397277 |       0.989974 |   0.967493    |                108438560 |
| R10        |   2.58474    |         0.335053 |     0           |  2.58853   |       0.333271 |   0           |                  1853984 |
| R60        |   1.16201    |         0.742499 |     0.0135012   |  1.17259   |       0.739419 |   0.0147331   |                 10148384 |
| R110       |   0.46489    |         0.907164 |     0.268643    |  0.482081  |       0.900545 |   0.263346    |                 18442784 |
| R160       |   0.169138   |         0.966494 |     0.6463      |  0.191704  |       0.958582 |   0.634745    |                 26737184 |
| R210       |   0.0628254  |         0.986835 |     0.85111     |  0.0894962 |       0.977839 |   0.831573    |                 35031584 |
| R260       |   0.027136   |         0.994296 |     0.934535    |  0.0556542 |       0.984932 |   0.910379    |                 43325984 |
| R310       |   0.0149479  |         0.996787 |     0.963986    |  0.0446103 |       0.987409 |   0.93693     |                 51620384 |
| R360       |   0.00906583 |         0.998063 |     0.978681    |  0.0395545 |       0.988783 |   0.951255    |                 59914784 |
| R410       |   0.00633389 |         0.998629 |     0.985996    |  0.0376929 |       0.98946  |   0.958026    |                 68209184 |
| R460       |   0.00489015 |         0.998923 |     0.989168    |  0.0370868 |       0.989833 |   0.962634    |                 76503584 |
| R510       |   0.00396089 |         0.999103 |     0.991334    |  0.036872  |       0.990129 |   0.965456    |                 84797984 |
| R560       |   0.00371487 |         0.999152 |     0.991931    |  0.0376033 |       0.990096 |   0.966083    |                 93092384 |
| R610       |   0.00355728 |         0.999173 |     0.992433    |  0.0383939 |       0.99008  |   0.966553    |                101386784 |
| R660       |   0.00340231 |         0.999189 |     0.992747    |  0.0390228 |       0.990043 |   0.967525    |                109681184 |
| R710       |   0.00337294 |         0.99918  |     0.992621    |  0.03984   |       0.989959 |   0.967556    |                117975584 |
| R760       |   0.00330195 |         0.999199 |     0.992935    |  0.0404524 |       0.989914 |   0.968151    |                126269984 |


## Model

Our model is a pre-LN variant of a BERT-style encoder using ReLU activations and additive learned positional embeddings.

| Parameter  | Value |
| ---------- | ----- |
| vocab_size | 32    |
| seq_len    | 32    |
| d_model    | 768   |
| n_heads    | 12    |
| d_ff       | 3072  |
| n_layers   | 12    |

[![PDF Preview](write-up/model-diagram.png)](write-up/model-diagram.pdf)

## Data

- Training data is from [Hugging face, agentlans/high-quality-english-sentences](https://huggingface.co/datasets/agentlans/high-quality-english-sentences)
- We convert all characters to lower case and remove all characters that are not alphabetic characters or a space
- We extract 32 characters from the start of each sentence (removing those not long enough)
- We encrypt each sentence using one random key shift
- We then train and test on approximately 32,000 $(\text{Encrypted text}, \text{Decrypted text})$ pairs

## Citations

- Alammar, Jay. ‘The Illustrated Transformer’. Accessed 1 September 2026. https://jalammar.github.io/illustrated-transformer/.
- Andrej Karpathy. Let’s Build GPT: From Scratch, in Code, Spelled Out. 2023. 1:56:19. https://www.youtube.com/watch?v=kCc8FmEb1nY.
- Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. ‘Layer Normalization’. arXiv:1607.06450. Preprint, arXiv, 21 July 2016. https://doi.org/10.48550/arXiv.1607.06450.
- Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. ‘BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding’. arXiv:1810.04805. Preprint, arXiv, 24 May 2019. https://doi.org/10.48550/arXiv.1810.04805.
- ‘The Annotated Transformer’. Accessed 1 September 2026. https://nlp.seas.harvard.edu/annotated-transformer/.
- Xiong, Ruibin, Yunchang Yang, Di He, et al. ‘On Layer Normalization in the Transformer Architecture’. arXiv:2002.04745. Preprint, arXiv, 29 June 2020. https://doi.org/10.48550/arXiv.2002.04745.
- YouTube. ‘3blue1brown - Deep Learning’. Accessed 30 August 2026. http://www.youtube.com/playlist?list=PLOH0RpNCcyWRxD8bYrbZbVZrto0U8axHR.
