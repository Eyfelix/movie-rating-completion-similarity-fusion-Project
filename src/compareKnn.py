import numpy as np
import pandas as pd
from numba import njit, prange
import time
import itertools



@njit(parallel=True, cache=True)
def _suir_eucl_numba(idxU, wU, idxI, wI, Csub, Msub):
    """Rectangle exact avec poids euclidien, pour chaque film t :
        num[t] = Σ_r Σ_c  w(a_r, b_c) · Csub[idxU[t,r], idxI[t,c]]   (cases notées seulement)
        den[t] = Σ_r Σ_c  w(a_r, b_c) · Msub[idxU[t,r], idxI[t,c]]
    avec w(a, b) = a·b / √(a² + b²) = 1 / √(1/a² + 1/b²)."""
    n, NU = idxU.shape
    NI = idxI.shape[1]
    num = np.zeros(n)
    den = np.zeros(n)
    for t in prange(n):
        s_num = 0.0
        s_den = 0.0
        for r in range(NU):
            a = np.float64(wU[t, r])
            if a <= 0.0:
                continue
            v = idxU[t, r]
            a2 = a * a
            for c in range(NI):
                b = np.float64(wI[t, c])
                if b <= 0.0:
                    continue
                j = idxI[t, c]
                if Msub[v, j] == 0.0:            # case sans note : contribution nulle
                    continue
                w = a * b / np.sqrt(a2 + b * b)
                s_num += w * Csub[v, j]
                s_den += w
        num[t] = s_num
        den[t] = s_den
    return num, den


def _suir_eucl(idxU, wU, idxI, wI, Csub, Msub):
    return _suir_eucl_numba(np.ascontiguousarray(idxU, dtype=np.int64),
                                np.ascontiguousarray(wU, dtype=np.float32),
                                np.ascontiguousarray(idxI, dtype=np.int64),
                                np.ascontiguousarray(wI, dtype=np.float32),
                                np.ascontiguousarray(Csub, dtype=np.float32),
                                np.ascontiguousarray(Msub, dtype=np.float32))


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

def _ratio(num, den):
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)

def sf_prepare(trainData, feats, reg_u=15.0, reg_i=10.0, n_iter=3, sim="paper"):
    """Tout ce qui ne dépend que du train : baseline, résidus, voisins utilisateurs triés,
    et les ingrédients des similarités films."""
    obs = ~np.isnan(trainData)
    M = obs.astype(np.float32)
    R0 = np.where(obs, trainData, 0.0)
    mu, b_u, b_i = regularized_baseline(trainData, obs, reg_i, reg_u, n_iter)
    C = np.where(obs, trainData - (mu + b_u[:, None] + b_i[None, :]), 0.0).astype(np.float32)

    # Similarités utilisateurs : cosinus sur notes brutes (papier) ou sur résidus (variante)
    Xu = R0 if sim == "paper" else C
    Xu = Xu / np.maximum(np.linalg.norm(Xu, axis=1, keepdims=True), 1e-9)
    S_U = Xu @ Xu.T
    np.fill_diagonal(S_U, -np.inf)
    ordU = np.argsort(-S_U, axis=1)[:, :-1]                  # voisins triés, soi-même exclu
    simU = np.take_along_axis(S_U, ordU, axis=1).astype(np.float32)

    years, G = feats
    prep = dict(sim=sim, M=M, C=C, mu=mu, b_u=b_u, b_i=b_i, ordU=ordU, simU=simU,
                years=years, G=G)
    if sim == "paper":
        # Cosinus ajusté (Sarwar) : notes centrées par la moyenne brute de l'utilisateur
        cnt = obs.sum(1)
        rbar = R0.sum(1) / np.maximum(cnt, 1)
        A = np.where(obs, trainData - rbar[:, None], 0.0).astype(np.float32)
        prep["A"], prep["A2"] = A, A ** 2
    else:
        Ct = C.T
        prep["Xn_items"] = (Ct / np.maximum(np.linalg.norm(Ct, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    return prep

def _sf_item_rows(prep, rows):
    """Ingrédients des similarités films pour les seuls films `rows` : 3 matrices (n_rows, I)."""
    M = prep["M"]
    if prep["sim"] == "paper":
        A, A2 = prep["A"], prep["A2"]
        num = A[:, rows].T @ A
        # normes calculées sur les seuls utilisateurs ayant noté les deux films
        den = np.sqrt((A2[:, rows].T @ M) * (M[:, rows].T @ A2))
        S_rating = _ratio(num, den)
    else:
        Xn = prep["Xn_items"]
        S_rating = Xn[rows] @ Xn.T
    years, G = prep["years"], prep["G"]
    inter = G[rows] @ G.T
    n = G.sum(1)
    union = n[rows, None] + n[None, :] - inter
    S_genre = _ratio(inter, union)
    D_year = np.abs(years[rows, None] - years[None, :])
    return S_rating.astype(np.float32), S_genre.astype(np.float32), D_year.astype(np.float32)

def _sf_item_sim(item_rows, rows, alpha, w_genre, tau, signed=False):
    """Similarité films finale. signed=False : partie positive seulement (papier).
    signed=True : similarités négatives conservées (KnnBoosted). Soi-même à 0."""
    S_rating, S_genre, D_year = item_rows
    if alpha == 1:
        S = S_rating.copy()
    else:
        S = alpha * S_rating + (1 - alpha) * (w_genre * S_genre + (1 - w_genre) * np.exp(-D_year / tau))
    if not signed:
        S = np.maximum(S, 0)
    S = S.astype(np.float32)
    S[np.arange(len(rows)), rows] = 0
    return S

@njit(parallel=True, cache=True)
def _suir_eucl_global(a, Crows, Mrows, idxI, wI):
    """Rectangle euclidien pour UN utilisateur et TOUS les films.
    a      : (NU,)      poids des voisins de u (fixes, indépendants du film)
    Crows  : (NU, I)    résidus de ces voisins ; Mrows : leur masque
    idxI   : (I, NI)    voisins globaux de chaque film ; wI : leurs poids
    Renvoie num, den de taille (I,)."""
    NU = a.shape[0]
    I, NI = idxI.shape
    num = np.zeros(I)
    den = np.zeros(I)
    for t in prange(I):
        s_num = 0.0
        s_den = 0.0
        for c in range(NI):
            b = np.float64(wI[t, c])
            if b == 0.0:
                continue
            j = idxI[t, c]
            b2 = b * b
            for r in range(NU):
                av = np.float64(a[r])
                if av == 0.0 or Mrows[r, j] == 0.0:
                    continue
                w = av * b / np.sqrt(av * av + b2)
                s_num += w * Crows[r, j]
                s_den += abs(w)
            num[t] = s_num
            den[t] = s_den
    return num, den

def _sur_neighbors(prep, u, i, Kmax, select="rated", signed=False, chunk=4096):
    """Voisins utilisateurs de chaque couple (u, i), triés par similarité décroissante.
    select="rated"  : les Kmax plus similaires PARMI ceux qui ont noté i (papier).
    select="global" : les Kmax plus similaires globalement (KnnBoosted) ; ceux qui n'ont pas
                      noté i restent dans la liste mais ne contribuent à rien (masque)."""
    M, ordU, simU = prep["M"], prep["ordU"], prep["simU"]
    n, K = len(u), min(Kmax, prep["ordU"].shape[1])
    idxU = np.zeros((n, Kmax), dtype=np.int64)
    wU = np.zeros((n, Kmax), dtype=np.float32)

    if select == "global":
        idxU[:, :K] = ordU[u, :K]
        w = simU[u, :K]
        wU[:, :K] = w if signed else np.maximum(w, 0)
        return idxU, wU

    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        nb, sim = ordU[u[sl]], simU[u[sl]]
        ok = M[nb, i[sl][:, None]] > 0
        if not signed:
            ok &= sim > 0
        rank = np.cumsum(ok, axis=1) - 1
        rr, cc = np.nonzero(ok & (rank < Kmax))
        idxU[s + rr, rank[rr, cc]] = nb[rr, cc]
        wU[s + rr, rank[rr, cc]] = sim[rr, cc]
    return idxU, wU

def _sir_neighbors(M, S, r, u, Kmax, select="rated"):
    """Voisins films de chaque couple, triés par similarité décroissante.
    select="rated"  : les Kmax plus similaires PARMI les films notés par u (papier).
    select="global" : les Kmax plus similaires globalement (KnnBoosted) ; calculé une fois
                      par film, puis partagé entre tous les couples de ce film."""
    n = len(u)
    idxI = np.zeros((n, Kmax), dtype=np.int64)
    wI = np.zeros((n, Kmax), dtype=np.float32)

    if select == "global":
        K = min(Kmax, S.shape[1])
        top = np.argpartition(-S, K - 1, axis=1)[:, :K]
        w = np.take_along_axis(S, top, 1)
        o = np.argsort(-w, axis=1)
        idxI[:, :K] = np.take_along_axis(top, o, 1)[r]
        wI[:, :K] = np.take_along_axis(w, o, 1)[r]
        return idxI, wI

    order = np.argsort(u, kind="stable")
    for grp in np.split(order, np.flatnonzero(np.diff(u[order])) + 1):
        J = np.flatnonzero(M[u[grp[0]]])
        if len(J) == 0:
            continue
        Ssub = S[np.ix_(r[grp], J)]
        K = min(Kmax, len(J))
        top = (np.argpartition(-Ssub, K - 1, axis=1)[:, :K] if K < len(J)
               else np.broadcast_to(np.arange(len(J)), Ssub.shape))
        w = np.take_along_axis(Ssub, top, 1)
        o = np.argsort(-w, axis=1)
        idxI[grp, :K] = J[np.take_along_axis(top, o, 1)]
        wI[grp, :K] = np.take_along_axis(w, o, 1)
    return idxI, wI

def _sf_parts(prep, u, i, idxU, wU, idxI, wI, ks_u, ks_i, kx_u=None, kx_i=None,
              budget=None, cross=True, max_elems=4_000_000):
    """Écarts SUR, SIR et SUIR pour toutes les tailles demandées.

    ks_u, ks_i : tailles des voisinages des termes directs.
    kx_u, kx_i : tailles du rectangle (défaut : ks_u, ks_i), prises au début des mêmes listes.
    budget     : si donné, on ne garde que les `budget` cases notées de plus fort poids
                 (logique « nombre de notes » du papier) ; sinon toutes comptent.

    sur : (n, len(ks_u))   sir : (n, len(ks_i))
    x_* : (n, len(kx_u) * len(kx_i ou budget)), index aplati via _sf_xcol."""
    C, M = prep["C"], prep["M"]
    ku, ki = np.asarray(ks_u, int), np.asarray(ks_i, int)
    xu = ku if kx_u is None else np.asarray(kx_u, int)
    xi = ki if kx_i is None else np.asarray(kx_i, int)
    n = len(u)

    mU = M[idxU, i[:, None]]                                  # 1 si le voisin a noté i
    mI = M[u[:, None], idxI]                                  # 1 si u a noté le film voisin
    numU = np.cumsum(wU * C[idxU, i[:, None]], 1)[:, ku - 1]
    denU = np.cumsum(np.abs(wU) * mU, 1)[:, ku - 1]
    numI = np.cumsum(wI * C[u[:, None], idxI], 1)[:, ki - 1]
    denI = np.cumsum(np.abs(wI) * mI, 1)[:, ki - 1]
    parts = {"sur": _ratio(numU, denU), "sur_ok": denU > 0,
             "sir": _ratio(numI, denI), "sir_ok": denI > 0}

    if cross:
        NU, NI = int(xu.max()), int(xi.max())
        second = xi if budget is None else np.asarray(budget, int)
        acc = {c: (np.zeros((n, len(xu) * len(second))),
                   np.zeros((n, len(xu) * len(second)))) for c in ("prod", "eucl")}
        b = max(1, int(max_elems // max(NU * NI, 1)))
        for s in range(0, n, b):
            sl = slice(s, min(s + b, n))
            nb_ = sl.stop - sl.start
            rows_, cols_ = idxU[sl, :NU][:, :, None], idxI[sl, :NI][:, None, :]
            Cb = C[rows_, cols_]
            Mb = M[rows_, cols_] * mU[sl, :NU][:, :, None] * mI[sl, :NI][:, None, :]
            a, c = wU[sl, :NU][:, :, None], wI[sl, :NI][:, None, :]
            P = a * c
            for name, W in (("prod", P),
                            ("eucl", P / np.sqrt(np.maximum(a * a + c * c, 1e-12)))):
                WV, WM = W * Cb * Mb, np.abs(W) * Mb
                if budget is None:
                    cv = WV.cumsum(1).cumsum(2)[:, xu - 1][:, :, xi - 1]
                    cm = WM.cumsum(1).cumsum(2)[:, xu - 1][:, :, xi - 1]
                    acc[name][0][sl] = cv.reshape(nb_, -1)
                    acc[name][1][sl] = cm.reshape(nb_, -1)
                else:
                    for ja, na in enumerate(xu):               # vivier na × xi[ja] -> ici na × na
                        nb_i = int(xi[min(ja, len(xi) - 1)])
                        Vn = WV[:, :na, :nb_i].reshape(nb_, -1)
                        Wn = WM[:, :na, :nb_i].reshape(nb_, -1)
                        o = np.argsort(-Wn, axis=1)            # poids décroissants
                        Vs = np.cumsum(np.take_along_axis(Vn, o, 1), axis=1)
                        Ws = np.cumsum(np.take_along_axis(Wn, o, 1), axis=1)
                        for jb, nbud in enumerate(second):
                            t = min(int(nbud), Vs.shape[1]) - 1
                            acc[name][0][sl, ja * len(second) + jb] = Vs[:, t]
                            acc[name][1][sl, ja * len(second) + jb] = Ws[:, t]
        for name, (num, den) in acc.items():
            parts[f"x_{name}"], parts[f"x_{name}_ok"] = _ratio(num, den), den > 0
    return parts

def _sf_xcol(ja, jb, n_second):
    """Index aplati de la colonne du rectangle."""
    return ja * n_second + jb

def _sf_combine(parts, ju, ji, jx, lam, delta, comb):
    """λ(1−δ)·SUR + (1−λ)(1−δ)·SIR + δ·SUIR, pour toutes les valeurs de λ et δ : (nλ, nδ, n)."""
    lam = np.asarray(lam, float)[:, None, None]
    delta = np.asarray(delta, float)[None, :, None]
    w1, w2, w3 = lam * (1 - delta), (1 - lam) * (1 - delta), delta + 0 * lam
    e1, e2 = parts["sur"][:, ju], parts["sir"][:, ji]
    e3 = parts[f"x_{comb}"][:, jx] if (comb and f"x_{comb}" in parts) else np.zeros_like(e1)
    return w1 * e1 + w2 * e2 + w3 * e3

@njit(parallel=True, cache=True)
def _suir_eucl_rows(nzU, aU, C, M, idxI, wI, out_num, out_den):
    """Rectangle euclidien pour UN utilisateur et TOUS les films, sans aucune copie.
    nzU : indices globaux des voisins de u ; aU : leurs poids.
    C, M : matrices complètes. idxI, wI : (I, NI) voisins globaux de chaque film."""
    I, NI = idxI.shape
    NU = nzU.shape[0]
    for t in prange(I):
        s_num = 0.0
        s_den = 0.0
        for c in range(NI):
            b = np.float64(wI[t, c])
            if b == 0.0:
                continue
            j = idxI[t, c]
            b2 = b * b
            for r in range(NU):
                v = nzU[r]
                if M[v, j] == 0.0:
                    continue
                a = np.float64(aU[r])
                w = a * b / np.sqrt(a * a + b2)
                s_num += w * C[v, j]
                s_den += abs(w)
        out_num[t] = s_num
        out_den[t] = s_den

@njit(parallel=True, cache=True)
def _suir_eucl_pairs(us, items, Su, C, M, idxI, wI):
    """Rectangle euclidien pour une liste de couples (u, i) seulement."""
    n = us.shape[0]
    NI = idxI.shape[1]
    U = Su.shape[1]
    num = np.zeros(n)
    den = np.zeros(n)
    for t in prange(n):
        u = us[t]
        i = items[t]
        s_num = 0.0
        s_den = 0.0
        for c in range(NI):
            b = np.float64(wI[i, c])
            if b == 0.0:
                continue
            j = idxI[i, c]
            b2 = b * b
            for v in range(U):
                a = np.float64(Su[u, v])
                if a == 0.0 or M[v, j] == 0.0:
                    continue
                w = a * b / np.sqrt(a * a + b2)
                s_num += w * C[v, j]
                s_den += abs(w)
        num[t] = s_num
        den[t] = s_den
    return num, den


def parse_movies(df):
    titles = df.iloc[:, 0].astype(str)
    genres = df.iloc[:, 1].astype(str)
    years = titles.str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    years = years.fillna(years.median())
    G = genres.str.get_dummies(sep="|")
    G = G.drop(columns=["(no genres listed)"], errors="ignore")
    return years.to_numpy(), G.to_numpy(dtype=float)
    
class SimilarityFusion:
    """Similarity fusion (Wang, de Vries & Reinders, SIGIR 2006) avec baseline régularisée
    et, en option, similarité de contenu pour les films."""

    def __init__(self, N=None, k_user=None, k_movie=None,
                 N_cross=None, k_user_cross=None, k_movie_cross=None,
                 select=None, lam=0.7, delta=0.7, comb="eucl", sim="paper", signed=False,
                 alpha=1.0, w_genre=0.7, tau=10.0, reg_u=0, reg_i=0, n_iter=1,
                 RMIN=0.5, RMAX=5.0, pair_chunk=20000):
        """N donné          -> logique papier : select='rated', budget en NOTES.
           k_user + k_movie -> logique KnnBoosted : select='global', budget en VOISINS.
           `select` permet de forcer l'un ou l'autre pour isoler les deux effets."""
        if (N is None) == (k_user is None and k_movie is None):
            raise ValueError("donner soit N, soit k_user ET k_movie")
        if N is None and (k_user is None or k_movie is None):
            raise ValueError("k_user et k_movie vont ensemble")
        self.paper_mode = N is not None
        self.select = select or ("rated" if self.paper_mode else "global")
        self.N, self.k_user, self.k_movie = N, k_user, k_movie
        self.N_cross, self.k_user_cross, self.k_movie_cross = N_cross, k_user_cross, k_movie_cross
        self.lam, self.delta, self.comb, self.sim, self.signed = lam, delta, comb, sim, signed
        self.alpha, self.w_genre, self.tau = alpha, w_genre, tau
        self.reg_u, self.reg_i, self.n_iter = reg_u, reg_i, n_iter
        self.RMIN, self.RMAX, self.pair_chunk = RMIN, RMAX, pair_chunk

    def _sizes(self):
        """(ku, ki, kxu, kxi, budget) effectifs, bornés par les données.
        Les tailles du rectangle ne peuvent pas dépasser celles des listes de voisins."""
        U1, I1 = self.prep["ordU"].shape[1], self.prep["M"].shape[1] - 1
        if self.paper_mode:
            ku = ki = int(min(self.N, U1, I1))
            kxu = kxi = int(min(self.N_cross or self.N, ku))
            return ku, ki, kxu, kxi, int(min(self.N_cross or self.N, kxu * kxi))
        ku, ki = int(min(self.k_user, U1)), int(min(self.k_movie, I1))
        kxu = int(min(self.k_user_cross or ku, ku))
        kxi = int(min(self.k_movie_cross or ki, ki))
        return ku, ki, kxu, kxi, None

    def predict_pairs(self, u, i):
        u, i = np.asarray(u), np.asarray(i)
        p = self.prep
        ku, ki, kxu, kxi, budget = self._sizes()
        out = np.empty(len(u))
        order = np.argsort(i, kind="stable")
        for s in range(0, len(u), self.pair_chunk):
            idx = order[s:s + self.pair_chunk]
            uu, ii = u[idx], i[idx]
            rows, r = np.unique(ii, return_inverse=True)
            S = _sf_item_sim(_sf_item_rows(p, rows), rows, self.alpha, self.w_genre, self.tau, self.signed)
            idxU, wU = _sur_neighbors(p, uu, ii, ku, self.select, self.signed)
            idxI, wI = _sir_neighbors(p["M"], S, r, uu, ki, self.select)
            parts = _sf_parts(p, uu, ii, idxU, wU, idxI, wI, [ku], [ki], [kxu], [kxi],
                              budget=None if budget is None else [budget],
                              cross=self.delta > 0)
            e = _sf_combine(parts, 0, 0, 0, [self.lam], [self.delta], self.comb)[0, 0]
            out[idx] = np.clip(p["mu"] + p["b_u"][uu] + p["b_i"][ii] + e, self.RMIN, self.RMAX)
        return out

    def fit(self, trainData, movieData):
        feats = parse_movies(pd.DataFrame(movieData))
        self.prep = sf_prepare(trainData, feats, self.reg_u, self.reg_i, self.n_iter, self.sim)
        return self



    def _item_sim_full(self, block=512):
        """Similarité films complète (I × I, float32), construite par blocs de lignes
        pour ne jamais garder plus d'un bloc d'ingrédients en mémoire."""
        p = self.prep
        I = p["M"].shape[1]
        S = np.empty((I, I), dtype=np.float32)
        for s in range(0, I, block):
            rows = np.arange(s, min(s + block, I))
            S[rows] = _sf_item_sim(_sf_item_rows(p, rows), rows, self.alpha, self.w_genre, self.tau, self.signed)
        return S

    def _predict_full_vectorized(self, verbose=True):
        """Matrice complète, vectorisée par utilisateur. Mode 'voisins' uniquement :
        en mode 'notes' la sélection des N meilleures cases ne se vectorise pas ainsi."""
        p = self.prep
        M, C, ordU, simU = p["M"], p["C"], p["ordU"], p["simU"]
        U, I = M.shape
        ku, ki, _, _, _ = self._sizes()
        # ku, ki = self._sizes()
        S = self._item_sim_full()
        cross = self.delta > 0
        eucl = cross and self.comb == "eucl"
        w1, w2, w3 = self.lam * (1 - self.delta), (1 - self.lam) * (1 - self.delta), self.delta
        zeros, no = np.zeros(I), np.zeros(I, dtype=bool)
        P = np.empty((U, I))
        t0 = time.time()

        for u in range(U):
            keep = simU[u] > 0
            nb, s = ordU[u][keep], simU[u][keep]
            Mo = M[nb]
            rank = np.cumsum(Mo, axis=0)
            selU = (Mo > 0) & (rank <= ku)                    # nu au lieu de N
            A = s[:, None] * selU
            numU, denU = (A * C[nb]).sum(0), A.sum(0)

            J = np.flatnonzero(M[u])
            K = min(ki, len(J))                               # ni au lieu de N
            if K > 0:
                SJ = S[:, J]
                top = (np.argpartition(-SJ, K - 1, axis=1)[:, :K] if K < len(J)
                       else np.broadcast_to(np.arange(len(J)), (I, len(J))))
                wI = np.take_along_axis(SJ, top, 1)
                numI, denI = (wI * C[u, J][top]).sum(1), wI.sum(1)
            else:
                numI, denI = zeros, zeros

            e1, a1 = _ratio(numU, denU), denU > 0
            e2, a2 = _ratio(numI, denI), denI > 0

            if cross and K > 0 and len(nb):
                Csub, Msub = C[np.ix_(nb, J)], M[np.ix_(nb, J)]
                if eucl:
                    vv, ii = np.nonzero(selU)
                    slot = rank[vv, ii].astype(np.int64) - 1
                    idxU = np.zeros((I, ku), dtype=np.int64)  # nu au lieu de N
                    wU = np.zeros((I, ku), dtype=np.float32)
                    idxU[ii, slot] = vv
                    wU[ii, slot] = s[vv]
                    numX, denX = _suir_eucl(idxU, wU, top, wI, Csub, Msub)
                else:
                    B = np.zeros((I, len(J)), dtype=np.float32)
                    np.put_along_axis(B, np.ascontiguousarray(top), wI, axis=1)
                    numX, denX = (A * (Csub @ B.T)).sum(0), (A * (Msub @ B.T)).sum(0)
                e3, a3 = _ratio(numX, denX), denX > 0
            else:
                e3, a3 = zeros, no

            P[u] = np.clip(p["mu"] + p["b_u"][u] + p["b_i"] + w1 * e1 + w2 * e2 + w3 * e3,
                           self.RMIN, self.RMAX)
            if verbose and (u + 1) % 100 == 0:
                print(f"  {u + 1}/{U} utilisateurs : {time.time() - t0:.0f}s")
        return P

    def _user_sim_matrix(self, k):
        """Similarités utilisateurs tronquées à k voisins globaux : (U, U)."""
        p = self.prep
        U = p["M"].shape[0]
        S = np.zeros((U, U), dtype=np.float32)
        np.put_along_axis(S, p["ordU"][:, :k], p["simU"][:, :k], axis=1)
        return S if self.signed else np.maximum(S, 0)

    def _item_neighbors(self, S, k):
        """(idxI, wI) : les k voisins de chaque film, par |similarité| décroissante."""
        k = min(k, S.shape[1])
        idxI = np.argpartition(-np.abs(S), k - 1, axis=1)[:, :k].astype(np.int64)
        wI = np.take_along_axis(S, idxI, axis=1).astype(np.float32)
        return np.ascontiguousarray(idxI), np.ascontiguousarray(wI)
    
    def _predict_global_vectorized(self, verbose=True):
        """Matrice complète en mode select='global' : les voisins ne dépendent pas du couple,
        donc tout se ramène à des produits matriciels (comme KnnBoosted)."""
        print("using global")
        p = self.prep
        M, C = p["M"], p["C"]
        ku, ki, kxu, kxi, _ = self._sizes()
        t0 = time.time()

        S_u = self._user_sim_matrix(ku)
        S_i = self._truncate(self._item_sim_full(), ki, self.signed)

        e1 = _ratio(S_u @ C, np.abs(S_u) @ M)
        e2 = _ratio(C @ S_i.T, M @ np.abs(S_i).T)
        if verbose:
            print(f"  termes directs : {time.time() - t0:.1f}s")

        if self.delta > 0:
            t0 = time.time()
            Sxu = S_u if kxu == ku else self._user_sim_matrix(kxu)
            Sxi = S_i if kxi == ki else self._truncate(S_i, kxi, self.signed)
            if self.comb == "prod":
                num = Sxu @ (C @ Sxi.T)                       # S_U · C · S_Iᵀ
                den = np.abs(Sxu) @ (M @ np.abs(Sxi).T)
            else:
                idxI, wI = self._item_neighbors(Sxi, kxi)
                num = np.zeros(C.shape)
                den = np.zeros(C.shape)
                for u in range(M.shape[0]):
                    nz = np.flatnonzero(Sxu[u])
                    if len(nz):
                        _suir_eucl_rows(nz, Sxu[u, nz], C, M, idxI, wI, num[u], den[u])
            e3 = _ratio(num, den)
            if verbose:
                print(f"  rectangle ({self.comb}) : {time.time() - t0:.1f}s")
        else:
            e3 = 0.0

        l, d = self.lam, self.delta
        e = l * (1 - d) * e1 + (1 - l) * (1 - d) * e2 + d * e3
        return np.clip(p["mu"] + p["b_u"][:, None] + p["b_i"][None, :] + e, self.RMIN, self.RMAX)
    def _predict_global_masked(self, mask, verbose=True):
        """Mode global restreint aux cases du masque : pour 'prod' les produits matriciels
        restent le plus rapide, pour 'eucl' on ne traite que les couples demandés."""
        p = self.prep
        M, C = p["M"], p["C"]
        ku, ki, kxu, kxi, _ = self._sizes()
        u, i = np.where(mask)

        S_u = self._user_sim_matrix(ku)
        S_i = self._truncate(self._item_sim_full(), ki, self.signed)
        e1 = _ratio((S_u @ C)[u, i], (np.abs(S_u) @ M)[u, i])
        e2 = _ratio((C @ S_i.T)[u, i], (M @ np.abs(S_i).T)[u, i])

        if self.delta > 0:
            Sxu = S_u if kxu == ku else self._user_sim_matrix(kxu)
            Sxi = S_i if kxi == ki else self._truncate(S_i, kxi, self.signed)
            if self.comb == "prod":
                e3 = _ratio((Sxu @ (C @ Sxi.T))[u, i],
                            (np.abs(Sxu) @ (M @ np.abs(Sxi).T))[u, i])
            else:
                idxI, wI = self._item_neighbors(Sxi, kxi)
                num, den = _suir_eucl_pairs(np.ascontiguousarray(u),
                                            np.ascontiguousarray(i),
                                            np.ascontiguousarray(Sxu),
                                            C, M, idxI, wI)
                e3 = _ratio(num, den)
        else:
            e3 = 0.0

        l, d = self.lam, self.delta
        e = l * (1 - d) * e1 + (1 - l) * (1 - d) * e2 + d * e3
        P = np.full(M.shape, np.nan)
        P[u, i] = np.clip(p["mu"] + p["b_u"][u] + p["b_i"][i] + e, self.RMIN, self.RMAX)
        return P
    
    @staticmethod
    def _truncate(S, k, signed):
        """Garde les k plus grandes valeurs de chaque ligne (le reste à 0)."""
        S = S.copy()
        if k < S.shape[1]:
            drop = np.argpartition(-S, k, axis=1)[:, k:]
            np.put_along_axis(S, drop, 0, axis=1)
        return S if signed else np.maximum(S, 0)
    
    def predict(self, mask=None, verbose=True):
        """Mode global : toujours vectorisé (produits matriciels), puis masqué si besoin.
        Mode papier : vectorisé si la configuration s'y prête, sinon par couples."""
        if not self.paper_mode and self.select == "global":
            if mask is not None:
                return self._predict_global_masked(mask, verbose)
            return self._predict_global_vectorized(verbose)

        ku, ki, kxu, kxi, budget = self._sizes()
        plain = (budget is None and kxu == ku and kxi == ki and not self.signed)
        if mask is None and plain:
            return self._predict_full_vectorized(verbose)
        if mask is None:
            if verbose:
                print("configuration non vectorisable : matrice complète par couples (lent).")
            u, i = np.indices(self.prep["M"].shape).reshape(2, -1)
        else:
            u, i = np.where(mask)
        P = np.full(self.prep["M"].shape, np.nan)
        P[u, i] = self.predict_pairs(u, i)
        return P




#* PARAMETRE A TESTER 
"""
paper_params = {"N":150, "lam":0.7, "delta":0.1} #* RMSE : 0.860
movieF_params = {"N":25, "lam":0.7, "delta":0.1, "alpha":0.3} #* RMSE : 0.859
kk_params = {"k_user":400, "k_movie":800, "delta":0.1, "alpha":0.3, "lam":0.7} #* RMSE : 0.856
kk_cross_params = {"k_user":400, "k_movie":900, "delta":0.1, "alpha":0.3, "lam":0.7, "k_user_cross":50, "k_movie_cross":400} #* RMSE : 0.855
residu_params = {"k_user":300, "k_movie":900, "delta":0.1, "alpha":0.3, "lam":0.4, "k_user_cross":100, "k_movie_cross":400, "sim":"residual"} #* RMSE : 0.867
simProd_params = {"k_user":300, "k_movie":900, "delta":0.1, "alpha":0.3, "lam":0.4, "k_user_cross":100, "k_movie_cross":400, "sim":"residual", "comb":"prod"} #* RMSE : 0.867
regu_params = {"k_user":500, "k_movie":900, "delta":0.1, "alpha":0.3, "lam":0.4, "k_user_cross":100, "k_movie_cross":400, "sim":"residual", "comb":"prod", "reg_u":10, "reg_i":10,"n_iter":5 } #* RMSE : 0.837
regu_params2 = {"k_user":500, "k_movie":900, "delta":0.1, "alpha":0.3, "lam":0.4, "k_user_cross":100, "k_movie_cross":400, "sim":"paper", "comb":"prod", "reg_u":10, "reg_i":10,"n_iter":5 } #* RMSE : 0.831
regu_params3 = {"k_user":500, "k_movie":900, "delta":0.1, "alpha":0.3, "lam":0.4, "k_user_cross":100, "k_movie_cross":400, "sim":"paper", "comb":"eucl", "reg_u":10, "reg_i":10,"n_iter":5 } #* RMSE : 0.831

"""


"""
#* Explication des paramètres 
#* Paramètre du papier
N :	Taille des voisinages SUR et SIR, et donc du rectangle (N × N)
lam	: Poids relatif SUR / SIR	Papier (λ, 0,7)
delta :	Poids du rectangle SUIR	Papier (δ, 0,7)

#* Mixte
comb :	"eucl" (papier) ou "prod" (ma version)	Change la manière dont sont combiner les notes du rectangle
sim	: "paper" (papier) ou "residual" (ma version)	Change le calcul de la similarité ("paper"==cosinus sur les notes) ("resudual"=cosinus sur les résidus r - (μ + b_u + b_i))


#! Ajouts de ma version Absente du papier
reg_u, reg_i, n_iter : Baseline régularisée
alpha, w_genre, tau	: Utilisation des features sur les films pour les films

"""