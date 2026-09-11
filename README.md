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

| Strategy | Train Loss | Train Char Acc | Train Seq Acc | Val Loss  | Val Char Acc | Val Seq Acc |
| -------- | ---------- | -------------- | ------------- | --------- | ------------ | ----------- |
| Full     | 0.00652514 | 0.998196       | 0.986436      | 0.0421419 | 0.989236     | 0.960942    |
| R10      | 2.56903    | 0.330807       | 0             | 2.57257   | 0.329191     | 0           |
| R100     | 0.491093   | 0.902265       | 0.252221      | 0.508335  | 0.896304     | 0.24924     |
| R200     | 0.0615376  | 0.987267       | 0.850419      | 0.0851891 | 0.979258     | 0.831729    |
| R300     | 0.0151412  | 0.996512       | 0.96182       | 0.0414493 | 0.988339     | 0.937212    |
| R500     | 0.00631302 | 0.998372       | 0.985306      | 0.0370988 | 0.989927     | 0.960158    |
| R600     | 0.00598713 | 0.998397       | 0.986938      | 0.0385707 | 0.989804     | 0.961036    |
| R700     | 0.00617911 | 0.998319       | 0.986844      | 0.0404629 | 0.989511     | 0.961224    |
| Energy95 | 0.00624856 | 0.998291       | 0.986593      | 0.0405921 | 0.989482     | 0.960879    |
| Energy90 | 0.00622869 | 0.998324       | 0.986562      | 0.039608  | 0.989551     | 0.96044     |
| Energy85 | 0.00629095 | 0.99829        | 0.985463      | 0.0383901 | 0.989779     | 0.960597    |
| Energy80 | 0.0069521  | 0.998142       | 0.983673      | 0.0383529 | 0.989624     | 0.958904    |
| Energy75 | 0.00767097 | 0.997938       | 0.981852      | 0.0380986 | 0.989545     | 0.957588    |
| Energy70 | 0.00838877 | 0.997797       | 0.978838      | 0.0380116 | 0.989404     | 0.955205    |
| Energy65 | 0.00994936 | 0.997402       | 0.974724      | 0.0386123 | 0.989085     | 0.951005    |
| Energy60 | 0.0125511  | 0.996804       | 0.968131      | 0.0404257 | 0.988551     | 0.94414     |
| Energy55 | 0.0163492  | 0.995901       | 0.957989      | 0.0428237 | 0.98778      | 0.934673    |
| Energy50 | 0.0237214  | 0.994198       | 0.937926      | 0.049075  | 0.986249     | 0.915614    |
| Q10      | 0.0586851  | 0.981165       | 0.88637       | 0.0905526 | 0.973772     | 0.871791    |
| K10      | 0.0997652  | 0.969778       | 0.838456      | 0.130395  | 0.962834     | 0.826024    |
| V10      | 0.0738639  | 0.977497       | 0.804986      | 0.123302  | 0.965632     | 0.780759    |
| Q100     | 0.00827258 | 0.997644       | 0.983108      | 0.0410541 | 0.989198     | 0.960377    |
| K100     | 0.00895643 | 0.997338       | 0.98226       | 0.0416287 | 0.989152     | 0.959813    |
| V100     | 0.0101268  | 0.997052       | 0.974222      | 0.0477865 | 0.987523     | 0.948904    |
| Q200     | 0.00721577 | 0.99796        | 0.984803      | 0.0414365 | 0.989271     | 0.961067    |
| K200     | 0.00719378 | 0.997969       | 0.984803      | 0.0412741 | 0.989312     | 0.960816    |
| V200     | 0.00783361 | 0.997743       | 0.981915      | 0.0445015 | 0.988535     | 0.956522    |

## Model

A Pre-LN variant of a BERT-style encoder using ReLU activations and additive learned positional embeddings.

| Parameter  | Value |
| ---------- | ----- |
| vocab_size | 32    |
| seq_len    | 32    |
| d_model    | 768   |
| n_heads    | 12    |
| d_ff       | 3072  |
| n_layers   | 12    |

[![PDF Preview](write-up/model-diagram.png)](write-up/model-diagram.pdf)

## Citeations

* Alammar, Jay. ‘The Illustrated Transformer’. Accessed 1 September 2026. https://jalammar.github.io/illustrated-transformer/.
* Andrej Karpathy. Let’s Build GPT: From Scratch, in Code, Spelled Out. 2023. 1:56:19. https://www.youtube.com/watch?v=kCc8FmEb1nY.
* Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. ‘Layer Normalization’. arXiv:1607.06450. Preprint, arXiv, 21 July 2016. https://doi.org/10.48550/arXiv.1607.06450.
* Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. ‘BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding’. arXiv:1810.04805. Preprint, arXiv, 24 May 2019. https://doi.org/10.48550/arXiv.1810.04805.
* ‘The Annotated Transformer’. Accessed 1 September 2026. https://nlp.seas.harvard.edu/annotated-transformer/.
* Xiong, Ruibin, Yunchang Yang, Di He, et al. ‘On Layer Normalization in the Transformer Architecture’. arXiv:2002.04745. Preprint, arXiv, 29 June 2020. https://doi.org/10.48550/arXiv.2002.04745.
* YouTube. ‘3blue1brown - Deep Learning’. Accessed 30 August 2026. http://www.youtube.com/playlist?list=PLOH0RpNCcyWRxD8bYrbZbVZrto0U8axHR.

