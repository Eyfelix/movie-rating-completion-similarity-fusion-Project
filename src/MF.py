import numpy as np
import pandas as pd
import itertools, time


class BiasedMF:
    """Factorisation de matrice avec biais, entraînée par moindres carrés alternés (ALS).
    r̂_ui = μ + b_u + b_i + p_u · q_i"""

    def __init__(self, n_factors=20, reg=0.1, reg_bias_u=15.0, reg_bias_i=10.0, weighted=True,
                 n_iter=15, init_std=0.1, seed=42, RMIN=0.5, RMAX=5.0,
                 row_chunk=1024, col_chunk=2048):
        self.n_factors, self.reg, self.weighted = n_factors, reg, weighted
        self.reg_bias_u, self.reg_bias_i = reg_bias_u, reg_bias_i
        self.n_iter, self.init_std, self.seed = n_iter, init_std, seed
        self.RMIN, self.RMAX = RMIN, RMAX
        self.row_chunk, self.col_chunk = row_chunk, col_chunk
        self.pred = None

    @staticmethod
    def _rmse(P, R, mask):
        return np.sqrt(np.mean((P[mask] - R[mask]) ** 2))

    def _solve_side(self, M, T, X, reg_bias):
        """Pour chaque ligne a, résout la ridge
            min_θ  Σ_j m_aj (T_aj − θ·X_j)²  +  reg_bias·θ₀²  +  reg·s_a·‖θ₁:‖²
        où X_j = [1, facteurs du côté fixé], θ = [biais, facteurs] de la ligne a,
        et s_a = nb de notes de a (ALS-WR) ou 1."""
        n_rows, n_cols = M.shape
        k = X.shape[1]
        n = M.sum(1)
        scale = np.maximum(n, 1.0) if self.weighted else np.ones(n_rows)
        B = (M * T) @ X                                   # second membre : Σ_j m_aj T_aj X_j
        theta = np.empty((n_rows, k))
        d = np.arange(1, k)

        for s in range(0, n_rows, self.row_chunk):
            sl = slice(s, min(s + self.row_chunk, n_rows))
            # A_a = Σ_j m_aj X_j X_jᵀ, calculé comme un produit matriciel avec les X_j X_jᵀ aplatis
            A = np.zeros((sl.stop - sl.start, k * k))
            for c in range(0, n_cols, self.col_chunk):
                cs = slice(c, min(c + self.col_chunk, n_cols))
                Xc = X[cs]
                A += M[sl, cs] @ (Xc[:, :, None] * Xc[:, None, :]).reshape(-1, k * k)
            A = A.reshape(-1, k, k)
            A[:, 0, 0] += reg_bias                                 # pénalité du biais
            A[:, d, d] += (self.reg * scale[sl])[:, None]          # pénalité des facteurs
            A[:, np.arange(k), np.arange(k)] += 1e-9               # stabilité numérique
            theta[sl] = np.linalg.solve(A, B[sl][..., None])[..., 0]   # toutes les ridges d'un coup
        return theta

    def _full(self):
        return np.clip(self.mu + self.b_u[:, None] + self.b_i[None, :] + self.P @ self.Q.T,
                       self.RMIN, self.RMAX)

    def fit(self, trainData, movieData=None, valData=None, verbose=False):
        """movieData est ignoré : même signature que KnnBoosted, pour le stacking.
        Si valData est fourni, la RMSE de validation est enregistrée à chaque itération."""
        rng = np.random.default_rng(self.seed)
        obs = ~np.isnan(trainData)
        M = obs.astype(float)
        MT = np.ascontiguousarray(M.T)
        R0 = np.where(obs, trainData, 0.0)
        U, I = trainData.shape

        self.mu = R0.sum() / M.sum()
        self.b_u, self.b_i = np.zeros(U), np.zeros(I)
        self.P = rng.normal(0, self.init_std, (U, self.n_factors))
        self.Q = rng.normal(0, self.init_std, (I, self.n_factors))
        val_mask = None if valData is None else ~np.isnan(valData)

        hist = []
        for it in range(1, self.n_iter + 1):
            # 1. Films fixés : une ridge par utilisateur sur [1, q_i] → (b_u, p_u)
            T = np.where(obs, R0 - self.mu - self.b_i[None, :], 0.0)
            th = self._solve_side(M, T, np.hstack([np.ones((I, 1)), self.Q]), self.reg_bias_u)
            self.b_u, self.P = th[:, 0], th[:, 1:]

            # 2. Utilisateurs fixés : une ridge par film sur [1, p_u] → (b_i, q_i)
            T = np.where(obs, R0 - self.mu - self.b_u[:, None], 0.0).T
            th = self._solve_side(MT, T, np.hstack([np.ones((U, 1)), self.P]), self.reg_bias_i)
            self.b_i, self.Q = th[:, 0], th[:, 1:]

            pred = self._full()
            row = {"iter": it, "RMSE_train": self._rmse(pred, trainData, obs)}
            if val_mask is not None:
                row["RMSE_val"] = self._rmse(pred, valData, val_mask)
            hist.append(row)
            if verbose:
                print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in row.items()))

        self.history = pd.DataFrame(hist)
        self.pred = pred
        return self

    def predict(self):
        return self.pred


def cv_mf(R, n_folds=5, seed=42, n_iter=20, verbose=True, **grid):
    """K-fold sur les notes de R, mêmes folds que cv_grid_search (même seed).
    grid : listes de valeurs pour les paramètres de BiasedMF (n_factors, reg, ...).
    Renvoie un tableau (une ligne par config × itération) avec la RMSE de chaque fold."""
    u, i = np.where(~np.isnan(R))
    folds = np.array_split(np.random.default_rng(seed).permutation(len(u)), n_folds)
    keys, combos = list(grid), list(itertools.product(*grid.values()))

    curves = []
    for f, idx in enumerate(folds):
        t0 = time.time()
        train, val = R.copy(), np.full_like(R, np.nan)
        train[u[idx], i[idx]] = np.nan
        val[u[idx], i[idx]] = R[u[idx], i[idx]]
        for combo in combos:
            params = dict(zip(keys, combo))
            m = BiasedMF(**params, n_iter=n_iter).fit(train, valData=val)
            curves.append(m.history.assign(fold=f, **params))
        if verbose:
            print(f"fold {f + 1}/{n_folds} : {time.time() - t0:.0f}s")

    hist = pd.concat(curves, ignore_index=True)
    res = hist.pivot_table(index=keys + ["iter"], columns="fold", values="RMSE_val")
    res.columns = [f"fold{c}" for c in res.columns]
    fold_cols = list(res.columns)
    res["RMSE"], res["std"] = res[fold_cols].mean(1), res[fold_cols].std(1)
    return res.reset_index().sort_values("RMSE").reset_index(drop=True), hist


if __name__ =="__main__":
    pass
    #* Exemple d'utilisation
    
    # mf = BiasedMF(n_factors=30, reg=0.04, n_iter=20).fit(trainData, valData=valData, verbose=True)
    # mf.history.plot(x="iter", y=["RMSE_train", "RMSE_val"], marker="o")