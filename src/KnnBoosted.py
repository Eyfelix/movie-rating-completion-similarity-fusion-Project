import numpy as np
import pandas as pd
from scipy import sparse

def observed(R):
    """Masque booléen des notes connues."""
    return ~np.isnan(R) & (R != 0)

def rmse(R_pred, R_true):
    """MSE calculée uniquement sur les notes connues de R_true."""
    m = observed(R_true)
    return np.sqrt(np.mean((R_pred[m] - R_true[m]) ** 2))

def user_means(R):
    """Moyenne de chaque utilisateur. Moyenne globale s'il n'a aucune note."""
    m = observed(R)
    Rf = np.where(m, R, 0.0)
    cnt = m.sum(1)
    glob = Rf.sum() / m.sum()
    return np.where(cnt > 0, Rf.sum(1) / np.maximum(cnt, 1), glob)

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


class KnnBoosted:
    def __init__(self, w_genre=0.7, tau=10.0, k_user=20, k_movie=20, lam=0.5, alpha=1.0,
                 gamma=0.0, k_user_cross=None, k_movie_cross=None, RMIN=0.5, RMAX=5.0, 
                 reg_i=10.0, reg_u=15.0, n_iter=3):
        self.w_genre, self.tau = w_genre, tau
        self.k_user, self.k_movie = k_user, k_movie
        self.lam, self.alpha, self.gamma = lam, alpha, gamma
        self.k_user_cross  = k_user  if k_user_cross  is None else k_user_cross
        self.k_movie_cross = k_movie if k_movie_cross is None else k_movie_cross
        self.RMIN, self.RMAX = RMIN, RMAX
        self.reg_i, self.reg_u, self.n_iter = reg_i, reg_u, n_iter

    # ---------- contenu des films (inchangé) ----------
    def parse_movies(self, df):
        titles = df.iloc[:, 0].astype(str)
        genres = df.iloc[:, 1].astype(str)
        years = titles.str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
        years = years.fillna(years.median())
        G = genres.str.get_dummies(sep="|")
        G = G.drop(columns=["(no genres listed)"], errors="ignore")
        return years.to_numpy(), G.to_numpy(dtype=float)

    def genre_similarity(self, G):
        inter = G @ G.T
        n = G.sum(1)
        union = n[:, None] + n[None, :] - inter
        return np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    def year_similarity(self, years, tau=10.0):
        return np.exp(-np.abs(years[:, None] - years[None, :]) / tau)

    def content_similarity(self, years, G, w_genre=0.7, tau=10.0):
        return w_genre * self.genre_similarity(G) + (1 - w_genre) * self.year_similarity(years, tau)

    # ---------- outils ----------

    @staticmethod
    def _cosine(C):
        """Similarité cosinus entre les lignes de C."""
        norms = np.linalg.norm(C, axis=1, keepdims=True)
        return (C @ C.T) / np.maximum(norms * norms.T, 1e-9)

    @staticmethod
    def _top_k(S, k):
        S = S.copy()
        np.fill_diagonal(S, -np.inf)          # soi-même ne peut pas être sélectionné
        if k < S.shape[0] - 1:
            drop = np.argpartition(-S, k, axis=1)[:, k:]
            np.put_along_axis(S, drop, 0, axis=1)
        np.fill_diagonal(S, 0)
        S = S.astype(np.float32)
        return S

    @staticmethod
    def _ratio(num, den):
        return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)

    # ---------- apprentissage ----------
    def fit(self, trainData, movieData):
        years, G = self.parse_movies(pd.DataFrame(movieData))
        self.S_content = self.content_similarity(years, G, self.w_genre, self.tau)

        self.observed = ~np.isnan(trainData)
        m = self.observed.astype(float)

        self.mu, self.b_u, self.b_i = regularized_baseline(
            trainData, self.observed, self.reg_i, self.reg_u, self.n_iter)
        self.baseline = self.mu + self.b_u[:, None] + self.b_i[None, :]     # (U, I)
        C = np.where(self.observed, trainData - self.baseline, 0.0) 

        # Similarités complètes, calculées une seule fois
        S_U_full = self._cosine(C)
        S_I_full = self.alpha * self._cosine(C.T) + (1 - self.alpha) * self.S_content

        # Termes directs
        self.S_user  = self._top_k(S_U_full, self.k_user)
        self.S_movie = self._top_k(S_I_full, self.k_movie)
        self.e_user  = self._ratio(self.S_user @ C,  np.abs(self.S_user) @ m)
        self.e_movie = self._ratio(C @ self.S_movie.T, m @ np.abs(self.S_movie).T)

        # Terme croisé : films proches chez les utilisateurs proches
        self.e_cross = None
        if self.gamma > 0:
            Su = sparse.csr_matrix(self._top_k(S_U_full, self.k_user_cross))
            Si = sparse.csr_matrix(self._top_k(S_I_full, self.k_movie_cross))
            num = (Si @ (Su @ C).T).T                  # = S_U · C · S_Iᵀ
            den = (abs(Si) @ (abs(Su) @ m).T).T        # = |S_U| · M · |S_I|ᵀ
            self.e_cross = self._ratio(num, den)

        self.pred = self.combine()
        return self

    def combine(self, lam=None, gamma=None):
        """Recombine les écarts déjà calculés, sans refaire le fit."""
        lam   = self.lam   if lam   is None else lam
        gamma = self.gamma if gamma is None else gamma
        e = lam * self.e_user + (1 - lam) * self.e_movie
        if gamma > 0:
            if self.e_cross is None:
                raise ValueError("Terme croisé non calculé : refaire le fit avec gamma > 0.")
            e = (1 - gamma) * e + gamma * self.e_cross
        return np.clip(self.baseline + e, self.RMIN, self.RMAX)

    def predict(self):
        return self.pred
        
if __name__ =="__main__":
        #* Les meilleurs paramètres trouvé pour chaque amélioration
        methodsParams = {
        "user knn": {'k_user': 340, 'w_genre': 0.5, 'tau': 15.0},
        "user + movie knn": {'k_user': 340, 'k_movie': 800, 'lam': 0.45, 'w_genre': 0.5, 'tau': 15.0,},
        "user + movie (featured) knn": {'k_user': 340, 'k_movie': 800, 'lam': 0.45, 'alpha': 0.9, 'w_genre': 0.5, 'tau': 15.0,},
        "user + movie rectangle": {'k_user': 340, 'k_movie': 800, 'lam': 0.35, 'gamma': 0.25, 'alpha': 0.88, 'w_genre': 0.5, 'tau': 15.0, 'k_user_cross': 65, 'k_movie_cross': 220},
        "user + movie rectangle regularized" :{'reg_i': 10.0, 'reg_u': 20.0, 'k_user': 250, 'k_movie': 1600, 'lam': 0.2, 'gamma': 0.3, 'alpha': 0.9, 'w_genre': 0.7, 'tau': 10.0, 'k_user_cross': 70, 'k_movie_cross': 150}
    }

    #* Les RMSE des différentes configuration, sur 90% train + val, 10% test
    # RMSE of user knn : 0.8522604666376097
    # RMSE of user + movie knn : 0.8349163761467014
    # RMSE of user + movie (featured) knn : 0.8330131136387208
    # RMSE of user + movie rectangle : 0.8269337494641137
    # RMSE of user + movie rectangle regularized : 0.825906022263758


#* Explication des paramètres 

"""
#* Calcul des knn 
k_user : Nombre de voisins de chaque utilisateur, choisis une fois pour toutes, quel que soit le film à prédire. Seuls ceux qui ont noté le film contribuent ensuite.
k_movie : Même chose pour les films : k voisins fixes par film, dont seuls ceux notés par l’utilisateur comptent.
k_user_cross, k_movie_cross	: Dimensions du rectangle, indépendantes des k directs. Ses lignes sont les voisins globaux de u et ses colonnes les voisins globaux de i, sans condition sur ce qu’ils ont noté.

#* Ajout des features pour les films
w_genre, tau : Composition de la similarité de contenu : w·Jaccard(genres) + (1−w)·exp(−|Δannée|/τ).
alpha : Mélange de la similarité entre films : α·cosinus des notes + (1−α)·similarité de contenu.

#* Mélange des knn
lam (λ) : Dosage entre le KNN utilisateurs (λ = 1) et le KNN films (λ = 0).
gamma (γ) : Part du terme croisé. γ = 0 : pas de rectangle.

#* Pour la régularisation
reg_u, reg_i : Pénalités L2 des biais utilisateur et film dans la baseline b_ui = μ + b_u + b_i. Chaque biais vaut n/(n + reg) fois le résidu moyen observé : peu de notes, biais tiré vers 0.
n_iter : Nombre d’allers-retours entre le calcul des b_i et celui des b_u. Les deux biais dépendent l’un de l’autre.


#* Autres
RMIN, RMAX : Bornes de la prédiction (clip).

"""