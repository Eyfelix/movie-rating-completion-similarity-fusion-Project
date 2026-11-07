"""Fusion KNN — modèle final.

Similarity fusion (Wang, de Vries & Reinders, SIGIR 2006) avec nos modifications :
voisinages de taille fixe (k_user / k_movie), rectangle dimensionné séparément,
similarité de contenu sur les films, combinaison par produit et baseline régularisée.

    r̂_ui = clip( b_ui + λ(1−δ)·e_U + (1−λ)(1−δ)·e_I + δ·e_X )

Entièrement vectorisé : le modèle se ramène à des produits matriciels. La similarité
entre films est calculée par blocs de lignes, tronquée, puis multipliée aussitôt :
la matrice I×I n'existe jamais en entier.
"""

import numpy as np
import pandas as pd

def parse_movies(df):
    titles = df.iloc[:, 0].astype(str)
    genres = df.iloc[:, 1].astype(str)
    years = titles.str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    years = years.fillna(years.median())
    G = genres.str.get_dummies(sep="|")
    G = G.drop(columns=["(no genres listed)"], errors="ignore")
    return years.to_numpy(), G.to_numpy(dtype=float)

def _ratio(num, den):
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)

def regularized_baseline(R, obs, reg_i=10.0, reg_u=15.0, n_iter=3):
    """Biais régularisés : minimise Σ (r_ui − μ − b_u − b_i)² + reg_u·Σ b_u² + reg_i·Σ b_i²."""
    M = obs.astype(float)
    Rf = np.where(obs, R, 0.0)
    mu = Rf.sum() / M.sum()
    n_i, n_u = M.sum(0), M.sum(1)                 # nb de notes par film / par utilisateur

    b_u, b_i = np.zeros(R.shape[0]), np.zeros(R.shape[1])
    for _ in range(n_iter):
        b_i = ((Rf - mu - b_u[:, None]) * M).sum(0) / np.maximum(reg_i + n_i, 1e-9)
        b_u = ((Rf - mu - b_i[None, :]) * M).sum(1) / np.maximum(reg_u + n_u, 1e-9)
    return mu, b_u, b_i

def _truncate_dense(S, k):
    """Copie de S où chaque ligne ne garde que ses k plus grandes valeurs."""
    S = S.copy()
    if k < S.shape[1]:
        drop = np.argpartition(-S, k, axis=1)[:, k:]
        np.put_along_axis(S, drop, 0.0, axis=1)
    return S


class FusionKNN:
    """Prédiction des notes manquantes par fusion de trois voisinages.

    Les valeurs par défaut sont les hyperparamètres retenus par validation croisée.
    """

    def __init__(self, k_user=500, k_movie=900, k_user_cross=100, k_movie_cross=400,
                 lam=0.4, delta=0.1, alpha=0.3, w_genre=0.7, tau=10.0,
                 reg_u=10.0, reg_i=10.0, n_iter=5,
                 RMIN=0.5, RMAX=5.0, item_block=512):
        self.k_user, self.k_movie = k_user, k_movie
        self.k_user_cross, self.k_movie_cross = k_user_cross, k_movie_cross
        self.lam, self.delta = lam, delta
        self.alpha, self.w_genre, self.tau = alpha, w_genre, tau
        self.reg_u, self.reg_i, self.n_iter = reg_u, reg_i, n_iter
        self.RMIN, self.RMAX = RMIN, RMAX
        self.item_block = item_block
        self.pred = None

    # ------------------------------------------------------------------ similarités

    @staticmethod
    def _user_similarity(R0):
        """Cosinus sur les notes brutes, partie positive, diagonale nulle. (U, U)"""
        X = R0 / np.maximum(np.linalg.norm(R0, axis=1, keepdims=True), 1e-9)
        S = np.maximum(X @ X.T, 0.0).astype(np.float32)
        np.fill_diagonal(S, 0.0)
        return S

    def _item_blocks(self, trainData, obs, M, feats):
        """Similarité entre films par blocs de lignes : cosinus ajusté (Sarwar) mélangé à
        la similarité de contenu, partie positive, diagonale nulle.
        Génère (rows, S_bloc) avec S_bloc de taille (len(rows), I), jamais la matrice I×I."""
        years, G = feats
        I = M.shape[1]

        # Cosinus ajusté : notes centrées par la moyenne brute de l'utilisateur
        cnt = np.maximum(obs.sum(1), 1)
        rbar = np.where(obs, trainData, 0.0).sum(1) / cnt
        A = np.where(obs, trainData - rbar[:, None], 0.0).astype(np.float32)
        A2 = A ** 2
        gsum = G.sum(1)

        for s in range(0, I, self.item_block):
            rows = np.arange(s, min(s + self.item_block, I))
            num = A[:, rows].T @ A
            # normes sur les seuls utilisateurs ayant noté les deux films
            den = np.sqrt((A2[:, rows].T @ M) * (M[:, rows].T @ A2))
            S = _ratio(num, den).astype(np.float32)

            if self.alpha < 1:
                inter = G[rows] @ G.T
                union = gsum[rows, None] + gsum[None, :] - inter
                jaccard = _ratio(inter, union).astype(np.float32)
                dy = np.abs(years[rows, None] - years[None, :]).astype(np.float32)
                year = np.exp(-dy / self.tau)
                content = self.w_genre * jaccard + (1 - self.w_genre) * year
                S = self.alpha * S + (1 - self.alpha) * content

            S = np.maximum(S, 0.0).astype(np.float32)
            S[np.arange(len(rows)), rows] = 0.0          # pas son propre voisin
            yield rows, S

    # ------------------------------------------------------------------ apprentissage

    def fit(self, trainData, movieData):
        obs = ~np.isnan(trainData)
        M = obs.astype(np.float32)
        R0 = np.where(obs, trainData, 0.0).astype(np.float32)
        U, I = trainData.shape
        k_i, k_xi = min(self.k_movie, I - 1), min(self.k_movie_cross, I - 1)

        # Baseline régularisée et résidus
        self.mu, self.b_u, self.b_i = regularized_baseline(
            trainData, obs, self.reg_i, self.reg_u, self.n_iter)
        self.baseline = self.mu + self.b_u[:, None] + self.b_i[None, :]
        C = np.where(obs, trainData - self.baseline, 0.0).astype(np.float32)

        # Voisinages utilisateurs (U×U, petit)
        S_u = _truncate_dense(self._user_similarity(R0), min(self.k_user, U - 1))
        S_xu = (S_u if self.k_user_cross >= self.k_user
                else _truncate_dense(S_u, min(self.k_user_cross, U - 1)))

        # Voisinages films, traités bloc par bloc : chaque bloc tronqué alimente
        # directement les produits matriciels (BLAS multithread), sans matrice creuse.
        num_i = np.empty((U, I), np.float32); den_i = np.empty((U, I), np.float32)
        cross = self.delta > 0
        if cross:
            T = np.empty((U, I), np.float32); Tm = np.empty((U, I), np.float32)
        feats = parse_movies(pd.DataFrame(movieData))
        for rows, S in self._item_blocks(trainData, obs, M, feats):
            S_i = _truncate_dense(S, k_i)
            num_i[:, rows], den_i[:, rows] = C @ S_i.T, M @ S_i.T
            if cross:
                S_xi = S_i if k_xi >= k_i else _truncate_dense(S_i, k_xi)
                T[:, rows], Tm[:, rows] = C @ S_xi.T, M @ S_xi.T

        # Écarts : moyennes pondérées des résidus, normalisées par le poids réellement utilisé
        e_u = _ratio(S_u @ C, S_u @ M)                       # mêmes notes, utilisateurs proches
        e_i = _ratio(num_i, den_i)                           # films proches, même utilisateur
        e_x = _ratio(S_xu @ T, S_xu @ Tm) if cross else 0.0  # rectangle : les deux à la fois

        w1 = self.lam * (1 - self.delta)
        w2 = (1 - self.lam) * (1 - self.delta)
        self.pred = np.clip(self.baseline + w1 * e_u + w2 * e_i + self.delta * e_x,
                            self.RMIN, self.RMAX)
        return self

    # ------------------------------------------------------------------ prédiction

    def predict(self, mask=None):
        """Matrice complète des prédictions, ou seulement les cases de `mask`."""
        if mask is None:
            return self.pred.copy()
        return np.where(mask, self.pred, np.nan)

    def predict_pairs(self, u, i):
        """Prédictions pour une liste de couples (utilisateur, film)."""
        return self.pred[np.asarray(u), np.asarray(i)]



bestParams = {"k_user":500, "k_movie":900, "delta":0.1, "alpha":0.3, "lam":0.4, "k_user_cross":100, "k_movie_cross":400, "reg_u":10, "reg_i":10,"n_iter":5 }