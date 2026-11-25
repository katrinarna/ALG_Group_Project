# Exact Nearest-Neighbour Search on Sentence Embeddings

This project compares **brute-force search** and **KD-trees** for exact k-nearest-neighbour (k-NN) search on real high-dimensional data, and measures how the KD-tree's performance depends on the number of dimensions.

The data are 50,000 English Wikipedia sentences embedded as 384-dimensional vectors with the `all-MiniLM-L6-v2` sentence-transformer model. Each of 500 held-out query sentences is used to find its 10 nearest neighbours (k = 10).

## Main results

| Method (n = 50,000, d = 384) | Recall@10 | Query time | Build time |
|---|---|---|---|
| Brute force | 1.0 | 34.1 ms | – |
| KD-tree (`scipy.spatial.KDTree`) | 1.0 | 42.3 ms | 0.45 s |

* At full dimension the KD-tree is exact but **slower than brute force** (≈ 0.8× speedup).
* When the vectors are truncated to fewer dimensions, the KD-tree is **35–40× faster at d = 2–4**, still faster at d = 8–16, and **slower than brute force from d = 32 upwards**.
* The break-even point (d ≈ 16) matches the rule of thumb that KD-trees only help when n ≫ 2^d (here n = 50,000 ≈ 2^15.6). This is the *curse of dimensionality*.

Full tables are in `wikipedia_nn_data/exact_search_results.csv` and `wikipedia_nn_data/dimensionality_results.csv`. Timings vary by a few milliseconds between runs.

## Repository structure

```
.
├── 01_data_preparation.ipynb         Sample sentences and compute embeddings
├── 02_brute_force_baseline.ipynb     Brute-force search, ground truth, baseline timing
├── 03_kd_tree.ipynb                  KD-tree vs. brute force at 384 dimensions
├── 04_dimensionality_analysis.ipynb  Both methods for d = 2 … 384
├── evaluation.py                     Shared recall@k and timing functions
└── wikipedia_nn_data/
    ├── sentences.json                50,000 database sentences
    ├── query_sentences.json          500 query sentences
    ├── embeddings.npy                Database embeddings (50000 × 384, float32)
    ├── queries.npy                   Query embeddings (500 × 384, float32)
    ├── ground_truth.npy              Exact top-10 neighbours of each query (500 × 10)
    ├── baseline_results.csv          Brute-force results (notebook 02)
    ├── exact_search_results.csv      Brute force + KD-tree results (notebook 03)
    └── dimensionality_results.csv    Results per dimension (notebook 04)
```

## How to run

Run the notebooks in order from the project folder (the data paths are relative to it):

1. **`01_data_preparation.ipynb`** *(optional)*: downloads the dataset from Hugging Face and encodes all sentences, which takes several minutes. The output files are already included in `wikipedia_nn_data/`, so this step can be skipped.
2. **`02_brute_force_baseline.ipynb`**: computes the ground truth and the brute-force baseline.
3. **`03_kd_tree.ipynb`**: builds and evaluates the KD-tree.
4. **`04_dimensionality_analysis.ipynb`**: repeats the comparison for different dimensions (takes a few minutes).

Notebooks 03 and 04 depend on files written by notebook 02.

## Requirements

Python 3.10. Versions used:

| Package | Version | Needed for |
|---|---|---|
| numpy | 2.2.3 | all notebooks |
| scipy | 1.15.2 | brute force (`cdist`), KD-tree |
| pandas | 2.2.3 | result tables |
| matplotlib | 3.10.1 | plots |
| tqdm | – | progress bars |
| jupyter | – | running the notebooks |
| datasets | 5.2.0 | notebook 01 only |
| sentence-transformers | 6.3.0 | notebook 01 only |

```bash
pip install numpy scipy pandas matplotlib tqdm jupyter
pip install datasets sentence-transformers   # only needed to re-run notebook 01
```

## Method notes

* **Distance.** Embeddings are normalised to unit length, so Euclidean distance ranks neighbours the same way as cosine similarity.
* **Ground truth.** The exact neighbours are computed with brute force. Recall@10 is the fraction of the true 10 neighbours that a method returns, averaged over the 500 queries.
* **Timing.** Average time per query over all 500 queries, answered one at a time, after one warm-up query; median of 3 repeats (`evaluation.py`).
* **Dimensionality experiment.** Dimensions are reduced by keeping the first *d* coordinates of each vector, and the ground truth is recomputed for every *d*. This changes the dimension while keeping the same points and queries; the truncated vectors are used only to measure how the algorithms scale, not as semantic embeddings.
