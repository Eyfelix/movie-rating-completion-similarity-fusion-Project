# Movie Rating Completion by Similarity Fusion

Predicting missing ratings in a sparse user × movie matrix. We start from a published collaborative filtering method (Wang et al., SIGIR 2006), then extend it with movie features, flexible neighbourhoods and a regularised baseline. **The final model beats Neural Collaborative Filtering on RMSE while running about 25× faster.**

> Team project (3 students) · M2 IASD Data Lab (PSL: ENS, Mines Paris, Dauphine) · October 2026
> 📄 [Report](reports/report.pdf) · 🔬 [Research notebook](search.ipynb)

## Results

All models are evaluated over 30 random train / validation / test splits (80 / 10 / 10) with fixed seeds.

| Model | Mean RMSE ↓ | Fit + predict time |
|---|---|---|
| User mean | ≈ 0.936 | < 0.1 s |
| ALS (matrix factorisation) | ≈ 0.895 | ≈ 14 s |
| Paper KNN (Wang et al., 2006) | 0.882 | – |
| + movie features | 0.881 | – |
| + flexible neighbourhoods | 0.879 | – |
| Neural Collaborative Filtering | ≈ 0.871 | ≈ 41 s |
| **Final KNN (+ regularised baseline)** | **≈ 0.856** | **≈ 1.6 s** |

**Takeaways**
- **Regularisation brings the main gain:** about −0.027 RMSE, more than all other changes combined. The final model is also the most stable across splits (std ≈ 0.007 vs ≈ 0.008).
- **A well-tuned neighbourhood method beats NCF here,** at a fraction of the cost, thanks to fully vectorised NumPy computation.
- **Movie features and flexible neighbourhoods help, but modestly** (about −0.001 and −0.003 RMSE).

## Method

### Baseline paper: similarity fusion

[Wang, de Vries & Reinders (SIGIR 2006)](https://doi.org/10.1145/1148170.1148257) predict a rating by fusing three sources of neighbouring ratings:
- **user-based:** similar users who rated the target movie;
- **item-based:** similar movies rated by the target user;
- **a "rectangle":** similar users' ratings of similar movies. Individually less reliable, but much more numerous, it acts as a smoothing background model in sparse matrices.

The three sources are mixed by two parameters, λ (user vs item side) and δ (weight of the rectangle).

### Our extensions

1. **Movie features.**
   - Genre similarity (Jaccard) and release-year similarity (exponential decay, bandwidth τ).
   - Blended with the rating-based similarity through a weight α. Content similarity is far less sparse than rating similarity (mean ≈ 0.2 vs ≈ 0.003).
2. **Fixed, decoupled neighbourhoods.**
   - Separate k for users and movies, since the matrix is far from square.
   - Neighbourhoods computed once and reused for every prediction, instead of rebuilt per rating as in the paper.
   - Two extra parameters bound the rectangle, removing mostly noisy cells.
3. **Product similarity in the rectangle.**
   - We replaced the paper's Euclidean combination with a plain product of user and movie similarities.
   - It is slightly better once the baseline is regularised, and it factorises, so the whole rectangle is computed as matrix products.
4. **Regularised baseline.**
   - The paper centres ratings on raw user and movie means, which are very noisy for movies with only a few ratings.
   - We estimate user and movie biases by penalised least squares, solved by alternating closed-form updates.
   - With a penalty of 10, a movie rated twice keeps 17% of its apparent bias, one rated 300 times keeps 97%.

### Final prediction

$$\hat r_{ui} = \mu + b_u + b_i + \lambda(1-\delta)\,e_U(u,i) + (1-\lambda)(1-\delta)\,e_I(u,i) + \delta\,e_X(u,i)$$

Here $e_U$, $e_I$ and $e_X$ are similarity-weighted averages of residuals from the three sources. The result is clipped to [0.5, 5].

Final hyperparameters, selected by cross-validation on the training matrix:

| $k_{user}$ | $k_{movie}$ | $cross_{user}$ | $cross_{movie}$ | α | λ | δ | $\beta_u = \beta_i$ |
|---|---|---|---|---|---|---|---|
| 500 | 900 | 100 | 400 | 0.3 | 0.4 | 0.1 | 10 |

## Repository structure

```
├── README.md
├── requirements.txt
├── search.ipynb     # Full research log: variants, ablations, hyperparameter search
├── report/          # Project report (PDF)
├── src/
│   ├── generate.py  # Entry point: completes a rating matrix
│   └── ...          # Model, similarities, baselines
├── datasets/        # Rating matrix and movie metadata (genres, years)
└── gridResult/      # Grid search results
```

## Quick start

```bash
pip install -r requirements.txt
python src/generate.py <path/to/matrix_to_complete>
```

## Limitations

- **Hand-tuned design.** Many hyperparameters (8 for the final model); the gains from features and neighbourhood tweaks are within one standard deviation across splits.
- **Ratings only.** No implicit feedback or temporal dynamics, both known to matter on rating data.
- **Ceiling of neighbourhood methods.** Further accuracy would likely require latent-factor or deep models, at a much higher compute cost.

## Authors

Félix de Batz · Léon Hogenmuller · Thanina Siouani
Supervised by Alexandre Verine (PSL)
