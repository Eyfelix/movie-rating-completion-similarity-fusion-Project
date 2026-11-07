"""
Améliorations du pipeline KNN : modèle de contenu par utilisateur, KNN sur ses résidus,
KNN à similarité implicite, factorisation bayésienne (BPMF) et blending par gradient boosting.

S'utilise avec ton code existant : KnnBoosted, regularized_baseline, oof_predictions,
evaluate_blend, blend_predict, rmse_vec. Adapte l'import ci-dessous au nom de ton fichier.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wishart
from sklearn.ensemble import HistGradientBoostingRegressor

from src.KnnBoosted import KnnBoosted, regularized_baseline        # <-- ton module

RMIN, RMAX = 0.5, 5.0
# =====================================================================================
# 1. Features des films : contenu + statistiques apprises sur le train
# =====================================================================================
def item_features(trainData, movieData, b_i):
    """Une ligne par film. Contient :
    - contenu   : genres (one-hot), année et année² (centrées-réduites), décennie (one-hot)
    - stats     : biais b_i, popularité log(1+n_i) centrée-réduite, et leurs produits/carrés
    Les stats sont recalculées sur chaque train (pas de fuite vers la validation)."""
    years, G = KnnBoosted().parse_movies(pd.DataFrame(movieData))
    yz = (years - years.mean()) / years.std()
    decade = pd.get_dummies(np.clip((years // 10) * 10, 1930, 2010).astype(int))
    n_i = (~np.isnan(trainData)).sum(0)
    lp = np.log1p(n_i)
    lp = (lp - lp.mean()) / lp.std()
    X = np.column_stack([G, yz, yz ** 2, decade.to_numpy(float),
                         b_i, lp, b_i ** 2, b_i * lp, lp ** 2])
    genres = pd.DataFrame(movieData).iloc[:, 1].astype(str).str.get_dummies(sep="|")
    genres = genres.drop(columns=["(no genres listed)"], errors="ignore").columns.tolist()
    names = (genres + ["année", "année²"] + [f"décennie {d}" for d in decade.columns]
             + ["b_i", "log_pop", "b_i²", "b_i·log_pop", "log_pop²"])
    return X, names


# =====================================================================================
# 2. Modèle de contenu : une ridge par utilisateur sur les features des films
# =====================================================================================
class UserContentRidge:
    """r̂_ui = μ + b_u + b_i + w_u · x_i
    w_u : préférences propres à l'utilisateur u (genres, époques, films populaires ou de niche,
    sensibilité à la « qualité » b_i du film). Une ridge fermée par utilisateur.
    Capte ce que le KNN ne voit pas : le KNN n'utilise le contenu qu'à travers la similarité."""

    def __init__(self, lam=20.0, reg_i=10.0, reg_u=15.0, n_iter=10, RMIN=0.5, RMAX=5.0):
        self.lam, self.reg_i, self.reg_u, self.n_iter = lam, reg_i, reg_u, n_iter
        self.RMIN, self.RMAX = RMIN, RMAX

    def fit(self, trainData, movieData):
        obs = ~np.isnan(trainData)
        self.mu, self.b_u, self.b_i = regularized_baseline(
            trainData, obs, self.reg_i, self.reg_u, self.n_iter)
        self.baseline = self.mu + self.b_u[:, None] + self.b_i[None, :]
        X, self.feature_names = item_features(trainData, movieData, self.b_i)
        C = np.where(obs, trainData - self.baseline, 0.0)
        d = X.shape[1]
        self.W = np.zeros((trainData.shape[0], d))
        for u in range(trainData.shape[0]):
            idx = np.flatnonzero(obs[u])
            if len(idx):
                Xu = X[idx]
                self.W[u] = np.linalg.solve(Xu.T @ Xu + self.lam * np.eye(d), Xu.T @ C[u, idx])
        self.X = X
        self.offset = self.baseline + self.W @ X.T          # non clippé : sert d'offset au KNN
        self.pred = np.clip(self.offset, self.RMIN, self.RMAX)
        return self

    def predict(self):
        return self.pred

    def preferences(self, u):
        """Préférences apprises de l'utilisateur u, triées (utile pour le rapport)."""
        return pd.Series(self.W[u], index=self.feature_names).sort_values()


# =====================================================================================
# 3. KNN sur les résidus d'un modèle (+ KNN films à similarité implicite)
# =====================================================================================
class KnnResidual(KnnBoosted):
    """Identique à KnnBoosted, sauf que :
    - les écarts moyennés sont r_ui − offset_ui (offset = prédiction d'un autre modèle,
      ex. UserContentRidge). offset=None → comportement d'origine (baseline) ;
    - les similarités restent calculées sur les résidus de la baseline (meilleur en pratique) ;
    - e_movie_bin : KNN films dont la similarité est le cosinus des colonnes du masque
      « a noté / n'a pas noté » (information implicite, dense et fiable)."""

    def __init__(self, k_movie_bin=400, **kw):
        super().__init__(**kw)
        self.k_movie_bin = k_movie_bin

    def fit(self, trainData, movieData, offset=None):
        years, G = self.parse_movies(pd.DataFrame(movieData))
        self.S_content = self.content_similarity(years, G, self.w_genre, self.tau)
        self.observed = ~np.isnan(trainData)
        m = self.observed.astype(float)

        self.mu, self.b_u, self.b_i = regularized_baseline(
            trainData, self.observed, self.reg_i, self.reg_u, self.n_iter)
        base = self.mu + self.b_u[:, None] + self.b_i[None, :]
        C_sim = np.where(self.observed, trainData - base, 0.0)
        self.baseline = base if offset is None else offset
        C = np.where(self.observed, trainData - self.baseline, 0.0)

        S_U_full = self._cosine(C_sim)
        S_I_full = self.alpha * self._cosine(C_sim.T) + (1 - self.alpha) * self.S_content

        self.S_user = self._top_k(S_U_full, self.k_user)
        self.S_movie = self._top_k(S_I_full, self.k_movie)
        self.e_user = self._ratio(self.S_user @ C, np.abs(self.S_user) @ m)
        self.e_movie = self._ratio(C @ self.S_movie.T, m @ np.abs(self.S_movie).T)

        S_bin = self._top_k(self._cosine(m.T), self.k_movie_bin)
        self.e_movie_bin = self._ratio(C @ S_bin.T, m @ S_bin.T)

        self.e_cross = None
        if self.gamma > 0:
            Su = sparse.csr_matrix(self._top_k(S_U_full, self.k_user_cross))
            Si = sparse.csr_matrix(self._top_k(S_I_full, self.k_movie_cross))
            num = (Si @ (Su @ C).T).T
            den = (abs(Si) @ (abs(Su) @ m).T).T
            self.e_cross = self._ratio(num, den)

        self.pred = self.combine()
        return self


# =====================================================================================
# 4. Factorisation bayésienne (BPMF, Salakhutdinov & Mnih 2008) sur les résidus d'un offset
# =====================================================================================
class BPMF:
    """Échantillonnage de Gibbs ; la prédiction est la moyenne des échantillons après burn-in.
    Plus robuste que l'ALS sur des données très creuses (pas de réglage fin de la régularisation)."""

    def __init__(self, n_factors=10, alpha=2.0, n_samples=120, burn_in=30, seed=0,
                 RMIN=0.5, RMAX=5.0):
        self.k, self.alpha, self.n_samples, self.burn_in = n_factors, alpha, n_samples, burn_in
        self.seed, self.RMIN, self.RMAX = seed, RMIN, RMAX

    @staticmethod
    def _hyper(X, rng, beta0=2.0):
        n, k = X.shape
        xbar, S = X.mean(0), np.cov(X, rowvar=False, bias=True)
        beta_n = beta0 + n
        Wn_inv = np.eye(k) + n * S + (beta0 * n / beta_n) * np.outer(xbar, xbar)
        Wn = np.linalg.inv((Wn_inv + Wn_inv.T) / 2)
        Lam = wishart.rvs(df=k + n, scale=Wn, random_state=rng)
        mu = rng.multivariate_normal(n * xbar / beta_n, np.linalg.inv(beta_n * Lam))
        return mu, Lam

    def _side(self, M, T, V, mu0, Lam0, rng):
        n, k = M.shape[0], V.shape[1]
        VV = (V[:, :, None] * V[:, None, :]).reshape(len(V), k * k)
        VV = VV.astype(np.float32)
        P = (M @ VV).astype(np.float32).reshape(n, k, k) * self.alpha + Lam0[None]
        b = self.alpha * (T @ V) + (Lam0 @ mu0)[None]
        L = np.linalg.cholesky(P)
        mean = np.linalg.solve(P, b[..., None])[..., 0]
        z = rng.standard_normal((n, k))
        return mean + np.linalg.solve(np.transpose(L, (0, 2, 1)), z[..., None])[..., 0]

    def fit(self, trainData, offset):
        rng = np.random.default_rng(self.seed)
        obs = ~np.isnan(trainData)
        M = obs.astype(float)
        T = np.where(obs, trainData - offset, 0.0)
        U, I = trainData.shape
        P = 0.1 * rng.standard_normal((U, self.k))
        Q = 0.1 * rng.standard_normal((I, self.k))
        acc = np.zeros((U, I))
        for it in range(self.burn_in + self.n_samples):
            P = self._side(M, T, Q, *self._hyper(P, rng), rng)
            Q = self._side(M.T, T.T, P, *self._hyper(Q, rng), rng)
            if it >= self.burn_in:
                acc += P @ Q.T
        self.E = acc / self.n_samples
        self.pred = np.clip(offset + self.E, self.RMIN, self.RMAX)
        return self

    def predict(self):
        return self.pred


# =====================================================================================
# 5. Fonctions « parts » pour ton oof_predictions / blend_predict
# =====================================================================================
KNN_PARAMS = {'k_user': 340, 'k_movie': 1000, 'lam': 0.2, 'gamma': 0.35, 'alpha': 0.88,
              'w_genre': 0.5, 'tau': 15.0, 'k_user_cross': 100, 'k_movie_cross': 300}
KNN_RES_PARAMS = {**KNN_PARAMS}


def knn_parts_v2(train, movieData, params=KNN_PARAMS, bpmf=True):
    """Ton KNN d'origine (autour de la baseline) + KNN implicite + BPMF."""
    m = KnnResidual(**params).fit(train, movieData)
    clip = lambda X: np.clip(X, m.RMIN, m.RMAX)
    parts = {"baseline": clip(m.baseline),
             "knn_user": m.combine(lam=1.0, gamma=0.0),
             "knn_movie": m.combine(lam=0.0, gamma=0.0),
             "knn_movie_bin": clip(m.baseline + m.e_movie_bin),
             "knn_best": m.predict()}
    if m.e_cross is not None:
        parts["knn_cross"] = clip(m.baseline + m.e_cross)
    if bpmf:
        parts["bpmf"] = BPMF(n_factors=20).fit(train, m.baseline).predict()
    return parts


def content_parts(train, movieData, params=KNN_RES_PARAMS, lam=20.0, bpmf=True):
    """Modèle de contenu, puis KNN / BPMF sur ses résidus."""
    cm = UserContentRidge(lam=lam).fit(train, movieData)
    k = KnnResidual(**params).fit(train, movieData, offset=cm.offset)
    clip = lambda X: np.clip(X, k.RMIN, k.RMAX)
    parts = {"content": cm.predict(),
             "c_knn_user": k.combine(lam=1.0, gamma=0.0),
             "c_knn_movie": k.combine(lam=0.0, gamma=0.0),
             "c_knn_movie_bin": clip(cm.offset + k.e_movie_bin),
             "c_knn_best": k.predict()}
    if k.e_cross is not None:
        parts["c_knn_cross"] = clip(cm.offset + k.e_cross)
    if bpmf:
        parts["c_bpmf"] = BPMF(n_factors=10).fit(train, cm.offset).predict()
    return parts


def meta_parts(train):
    """Méta-features pour le blender non linéaire : le meilleur mélange dépend
    du nombre de notes de l'utilisateur et du film, et de leur dispersion."""
    obs = ~np.isnan(train)
    n_u, n_i = obs.sum(1), obs.sum(0)
    with np.errstate(all="ignore"):
        sd_u, sd_i = np.nanstd(train, 1), np.nanstd(train, 0)
    sd_u = np.nan_to_num(sd_u, nan=np.nanmean(sd_u))
    sd_i = np.nan_to_num(sd_i, nan=np.nanmean(sd_u))
    U, I = train.shape
    return {"log_nu": np.repeat(np.log1p(n_u)[:, None], I, 1),
            "log_ni": np.repeat(np.log1p(n_i)[None, :], U, 0),
            "sd_u": np.repeat(sd_u[:, None], I, 1),
            "sd_i": np.repeat(sd_i[None, :], U, 0)}


# =====================================================================================
# 6. Blender par gradient boosting
#    cible = note − baseline ; features = écarts (prédiction − baseline) + méta-features.
#    Donner les écarts plutôt que les prédictions brutes aide nettement les arbres.
# =====================================================================================
GBM_PARAMS = dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=31,
                  min_samples_leaf=200, l2_regularization=1.0, random_state=0)


def _gbm_X(get, features, meta, base):
    b = get(base)
    return np.column_stack([get(n) - b for n in features] + [get(n) for n in meta])


def evaluate_blend_gbm(oof, features, meta=(), base="baseline", **gbm_params):
    """Même protocole que evaluate_blend : blender entraîné sur 4 folds d'OOF, testé sur le 5e."""
    params = {**GBM_PARAMS, **gbm_params}
    rows = []
    for f in sorted(oof.fold.unique()):
        tr, te = oof[oof.fold != f], oof[oof.fold == f]
        g = HistGradientBoostingRegressor(**params).fit(
            _gbm_X(lambda n: tr[n].to_numpy(), features, meta, base), (tr.note - tr[base]).to_numpy())
        p = te[base].to_numpy() + g.predict(_gbm_X(lambda n: te[n].to_numpy(), features, meta, base))
        rows.append({"fold": f, "blend_gbm": float(np.sqrt(np.mean((np.clip(p, 0.5, 5.0) - te.note) ** 2)))})
    return pd.DataFrame(rows).set_index("fold")


def fit_gbm_blender(oof, features, meta=(), base="baseline", **gbm_params):
    return HistGradientBoostingRegressor(**{**GBM_PARAMS, **gbm_params}).fit(
        _gbm_X(lambda n: oof[n].to_numpy(), features, meta, base), (oof.note - oof[base]).to_numpy())


def blend_predict_gbm(models, gbm, features, meta, R_train, base="baseline",
                      RMIN=0.5, RMAX=5.0, chunk=50):
    """Refit des modèles sur R_train puis prédiction de la matrice complète, par blocs d'utilisateurs."""
    parts = {}
    for fit_parts in models.values():
        parts.update(fit_parts(R_train))
    U, I = R_train.shape
    P = np.empty((U, I))
    for s in range(0, U, chunk):
        sl = slice(s, min(s + chunk, U))
        X = _gbm_X(lambda n: parts[n][sl].ravel(), features, meta, base)
        P[sl] = parts[base][sl] + gbm.predict(X).reshape(-1, I)
    return np.clip(P, RMIN, RMAX)

"""KnnShared : similarités calculées une fois, parts produites pour plusieurs offsets."""
class KnnShared(KnnBoosted):
    """fit() calcule seulement les similarités complètes (users, films, implicite).
    parts_for(offset, params) renvoie les parts KNN autour de `offset` avec les k/lam/gamma de params."""

    def fit(self, trainData, movieData, w_genre=0.5, tau=15.0, alpha=0.88,
            reg_i=10.0, reg_u=15.0, n_iter=10):
        years, G = self.parse_movies(pd.DataFrame(movieData))
        self.observed = ~np.isnan(trainData)
        self.m = self.observed.astype(float)
        self.train = trainData
        mu, b_u, b_i = regularized_baseline(trainData, self.observed, reg_i, reg_u, n_iter)
        self.baseline = mu + b_u[:, None] + b_i[None, :]
        C = np.where(self.observed, trainData - self.baseline, 0.0)
        self.S_U_full = self._cosine(C)
        self.S_I_full = alpha * self._cosine(C.T) + (1 - alpha) * self.content_similarity(years, G, w_genre, tau)
        self.S_B_full = self._cosine(self.m.T)
        return self

    def parts_for(self, offset, p, prefix=""):
        m = self.m
        C = np.where(self.observed, self.train - offset, 0.0)
        C, m = C.astype(np.float32), m.astype(np.float32)
        clip = lambda X: np.clip(X, RMIN, RMAX)
        S = self._top_k(self.S_U_full, p["k_user"]);  e_user = self._ratio(S @ C, np.abs(S) @ m)
        S = self._top_k(self.S_I_full, p["k_movie"]); e_movie = self._ratio(C @ S.T, m @ np.abs(S).T)
        S = self._top_k(self.S_B_full, p.get("k_movie_bin", 400)); e_bin = self._ratio(C @ S.T, m @ S.T)
        e = p["lam"] * e_user + (1 - p["lam"]) * e_movie
        parts = {prefix + "knn_user": clip(offset + e_user), prefix + "knn_movie": clip(offset + e_movie),
                 prefix + "knn_movie_bin": clip(offset + e_bin)}
        if p.get("gamma", 0) > 0:
            Su = sparse.csr_matrix(self._top_k(self.S_U_full, p.get("k_user_cross", p["k_user"])))
            Si = sparse.csr_matrix(self._top_k(self.S_I_full, p.get("k_movie_cross", p["k_movie"])))
            e_cross = self._ratio((Si @ (Su @ C).T).T, (abs(Si) @ (abs(Su) @ m).T).T)
            parts[prefix + "knn_cross"] = clip(offset + e_cross)
            e = (1 - p["gamma"]) * e + p["gamma"] * e_cross
        parts[prefix + "knn_best"] = clip(offset + e)
        return parts


def make_parts_shared(train, movieData, knn_base=KNN_PARAMS, knn_content=KNN_RES_PARAMS, lam_c=20.0,
                      bpmf_base=None, bpmf_content=None, sim=None):
    """Toutes les parts (baseline + contenu) avec un seul calcul de similarités.
    sim : dict w_genre/tau/alpha/reg_i/reg_u pour les similarités (défaut : ceux de knn_base)."""
    sim = sim or {k: knn_base[k] for k in ("w_genre", "tau", "alpha", "reg_i", "reg_u") if k in knn_base}
    k = KnnShared().fit(train, movieData, **sim)
    parts = {"baseline": np.clip(k.baseline, RMIN, RMAX)}
    parts.update(k.parts_for(k.baseline, knn_base))
    if bpmf_base is not None:
        parts["bpmf"] = BPMF(**bpmf_base).fit(train, k.baseline).predict()
    if knn_content is not None:
        cm = UserContentRidge(lam=lam_c).fit(train, movieData)
        parts["content"] = cm.predict()
        parts.update(k.parts_for(cm.offset, knn_content, prefix="c_"))
        if bpmf_content is not None:
            parts["c_bpmf"] = BPMF(**bpmf_content).fit(train, cm.offset).predict()
    return parts