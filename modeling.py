from __future__ import annotations

import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.stats import chi2

#  MANUAL MULTINOMIAL NAIVE BAYES

class ManualMultinomialNB:

    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError(f"alpha harus > 0, dapat: {alpha}")
        self.alpha              = alpha
        self.classes_           = None   # array label unik
        self.class_log_prior_   = None   # dict {kelas: log P(kelas)}
        self.feature_log_prob_  = None   # dict {kelas: array log P(kata|kelas)}
        self.n_class_samples_   = None   # dict {kelas: jumlah dokumen latih}
        self.n_features_        = None   # jumlah fitur (|V|)

    def fit(self, X, y):
       
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)
        y = np.array(y)

        self.classes_    = np.unique(y)
        self.n_features_ = X.shape[1]
        n_docs           = len(y)

        class_count   = {}
        feature_count = {}

        for c in self.classes_:
            mask        = (y == c)
            X_c         = X[mask]
            class_count[c]   = X_c.shape[0]
            # Jumlah total bobot TF-IDF per fitur untuk kelas c
            # np.asarray().flatten() mengkonversi matrix 1×n ke array 1D
            feature_count[c] = np.asarray(X_c.sum(axis=0)).flatten()

        # Log prior: log P(kelas) = log(n_kelas / n_total)
        self.class_log_prior_ = {
            c: np.log(class_count[c] / n_docs)
            for c in self.classes_
        }

        # Log likelihood dengan Laplace smoothing:
        # log P(kata | kelas) = log((count(kata,kelas) + α) / (Σ_count + α×|V|))
        self.feature_log_prob_ = {}
        for c in self.classes_:
            smoothed    = feature_count[c] + self.alpha
            self.feature_log_prob_[c] = np.log(smoothed / smoothed.sum())

        self.n_class_samples_ = {c: class_count[c] for c in self.classes_}

        return self

    def predict(self, X):
        
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)

        predictions = []
        for i in range(X.shape[0]):
            x      = X[i]
            scores = {
                c: self.class_log_prior_[c]
                   + x.dot(self.feature_log_prob_[c]).item()
                for c in self.classes_
            }
            predictions.append(max(scores, key=scores.get))

        return np.array(predictions)

    def predict_proba(self, X):
        
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)

        probs_all = []
        for i in range(X.shape[0]):
            x   = X[i]
            jll = np.array([
                self.class_log_prior_[c] + x.dot(self.feature_log_prob_[c]).item()
                for c in self.classes_
            ])
            # Log-sum-exp untuk stabilitas numerik
            jll_max  = np.max(jll)
            exp_jll  = np.exp(jll - jll_max)
            probs_all.append(exp_jll / exp_jll.sum())

        return np.array(probs_all)

    def get_top_features(self, class_label, feature_names, top_n=10):
        
        if class_label not in self.feature_log_prob_:
            raise ValueError(f"Kelas '{class_label}' tidak ditemukan di model.")
        log_probs = self.feature_log_prob_[class_label]
        top_idx   = np.argsort(log_probs)[-top_n:][::-1]
        return [(feature_names[i], float(log_probs[i])) for i in top_idx]


#  MANUAL K-NEAREST NEIGHBORS

class ManualKNN:

    VALID_METRICS = ("cosine", "euclidean", "manhattan")

    def __init__(self, k: int = 5, metric: str = "cosine"):
        if metric not in self.VALID_METRICS:
            raise ValueError(
                f"metric harus salah satu dari {self.VALID_METRICS}, dapat: '{metric}'"
            )
        if k < 1:
            raise ValueError(f"k harus >= 1, dapat: {k}")

        self.k              = k
        self.metric         = metric
        self.X_train        = None   # csr_matrix data latih
        self.y_train        = None   # array label data latih
        self.X_train_norm   = None   # precomputed L2 norm (untuk cosine & euclidean)
        self.classes_       = None   # array label unik

    def fit(self, X, y):
        
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)
        self.X_train      = X
        self.y_train      = np.array(y)
        self.classes_     = np.unique(self.y_train)
        # Precompute ‖x‖ untuk setiap dokumen latih (dipakai cosine & euclidean)
        self.X_train_norm = np.sqrt(X.multiply(X).sum(axis=1)).A1
        return self

    # Metrik kemiripan

    def _cosine_similarity(self, x: csr_matrix) -> np.ndarray:
       
        dot   = self.X_train.dot(x.T).toarray().flatten()
        x_norm = float(np.sqrt(x.multiply(x).sum()))
        denom  = self.X_train_norm * x_norm
        denom[denom == 0] = 1e-10  # hindari pembagian nol (dokumen kosong)
        return dot / denom

    def _euclidean_similarity(self, x: csr_matrix) -> np.ndarray:
        
        x_norm_sq = float(x.multiply(x).sum())
        dot       = self.X_train.dot(x.T).toarray().flatten()
        dist_sq   = self.X_train_norm ** 2 + x_norm_sq - 2.0 * dot
        dist_sq   = np.clip(dist_sq, 0.0, None)
        return 1.0 / (1.0 + np.sqrt(dist_sq))

    def _manhattan_similarity(self, x: csr_matrix) -> np.ndarray:
        
        MAX_CELLS = 50_000_000
        n_cells   = self.X_train.shape[0] * self.X_train.shape[1]
        if n_cells > MAX_CELLS:
            raise MemoryError(
                f"Dataset terlalu besar untuk Manhattan distance "
                f"({n_cells:,} sel > batas {MAX_CELLS:,}). "
                "Gunakan metric='cosine' atau perkecil dataset."
            )
        x_dense       = x.toarray().flatten()
        X_train_dense = self.X_train.toarray()
        dist          = np.abs(X_train_dense - x_dense).sum(axis=1)
        return 1.0 / (1.0 + dist)

    def _compute_similarity(self, x: csr_matrix) -> np.ndarray:
        """Dispatch ke metrik yang dipilih."""
        if self.metric == "cosine":
            return self._cosine_similarity(x)
        elif self.metric == "euclidean":
            return self._euclidean_similarity(x)
        elif self.metric == "manhattan":
            return self._manhattan_similarity(x)

    #Prediksi

    def predict(self, X_test) -> np.ndarray:
       
        if not isinstance(X_test, csr_matrix):
            X_test = csr_matrix(X_test)

        predictions = []
        for i in range(X_test.shape[0]):
            x            = X_test[i]
            sims         = self._compute_similarity(x)
            top_k_idx    = np.argsort(sims)[-self.k:]
            top_k_labels = self.y_train[top_k_idx]
            top_k_sims   = sims[top_k_idx]

            votes = defaultdict(float)
            for lbl, sim in zip(top_k_labels, top_k_sims):
                votes[lbl] += sim

            predictions.append(max(votes, key=votes.get))

        return np.array(predictions)

    def predict_proba(self, X_test) -> np.ndarray:
        
        if not isinstance(X_test, csr_matrix):
            X_test = csr_matrix(X_test)

        probs_all = []
        for i in range(X_test.shape[0]):
            x            = X_test[i]
            sims         = self._compute_similarity(x)
            top_k_idx    = np.argsort(sims)[-self.k:]
            top_k_labels = self.y_train[top_k_idx]
            top_k_sims   = sims[top_k_idx]

            weight_dict  = {cls: 0.0 for cls in self.classes_}
            total_weight = top_k_sims.sum()

            for lbl, sim in zip(top_k_labels, top_k_sims):
                weight_dict[lbl] += sim

            if total_weight == 0:
                probs = [1.0 / len(self.classes_)] * len(self.classes_)
            else:
                probs = [weight_dict[cls] / total_weight for cls in self.classes_]

            probs_all.append(probs)

        return np.array(probs_all)

    def get_neighbors(self, x, feature_names=None):
       
        if not isinstance(x, csr_matrix):
            x = csr_matrix(x)

        sims      = self._compute_similarity(x)
        top_k_idx = np.argsort(sims)[-self.k:][::-1]  # descending

        result = []
        for rank, idx in enumerate(top_k_idx, start=1):
            entry = {
                "rank":       rank,
                "train_idx":  int(idx),
                "label":      self.y_train[idx],
                "similarity": float(sims[idx]),
            }
            if feature_names is not None:
                # Fitur yang bernilai > 0 di kedua dokumen
                x_arr   = x.toarray().flatten()
                tr_arr  = self.X_train[idx].toarray().flatten()
                shared  = np.where((x_arr > 0) & (tr_arr > 0))[0]
                shared_sorted = shared[np.argsort(x_arr[shared])[::-1]][:5]
                entry["top_shared_features"] = [
                    feature_names[i] for i in shared_sorted
                ]
            result.append(entry)

        return result

#  PENCARIAN K OPTIMAL

def knn_search_k(
    X_train, y_train, X_val, y_val,
    k_list: list[int] | None = None,
) -> dict:
    
    from evaluation import accuracy_score_manual, classification_report_manual

    if k_list is None:
        k_list = [1, 3, 5, 7, 9]

    results   = []
    f1_list   = []
    acc_list  = []
    best_k    = k_list[0]
    best_f1   = -1.0
    best_acc  = 0.0

    for k in k_list:
        knn    = ManualKNN(k=k, metric="cosine")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)

        acc        = float(accuracy_score_manual(y_val, y_pred))
        rep, _, _  = classification_report_manual(y_val, y_pred)
        macro_f1   = float(rep["macro_avg"]["f1"])

        results.append({"k": k, "accuracy": acc, "macro_f1": macro_f1})
        f1_list.append(macro_f1)
        acc_list.append(acc)

        if macro_f1 > best_f1:
            best_f1  = macro_f1
            best_k   = k
            best_acc = acc

    return {
        "results":  results,
        "best_k":   best_k,
        "best_f1":  best_f1,
        "best_acc": best_acc,
        "k_list":   k_list,
        "f1_list":  f1_list,
        "acc_list": acc_list,
    }


def nb_search_alpha(
    X_train, y_train, X_val, y_val,
    alpha_list: list[float],
    classification_report_fn,
) -> list[dict]:
    
    results = []
    for alpha in alpha_list:
        nb     = ManualMultinomialNB(alpha=alpha)
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_val)
        rep, _, _ = classification_report_fn(y_val, y_pred)
        results.append({
            "alpha":    float(alpha),
            "macro_f1": float(rep["macro_avg"]["f1"]),
        })
    return results


def knn_compare_metrics(
    X_train, y_train, X_val, y_val,
    k: int,
    classification_report_fn,
    metrics: tuple = ("cosine", "euclidean", "manhattan"),
) -> list[dict]:
    
    from evaluation import accuracy_score_manual

    results = []
    for metric in metrics:
        try:
            knn = ManualKNN(k=k, metric=metric)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)
            rep, _, _ = classification_report_fn(y_val, y_pred)
            results.append({
                "metric":   metric,
                "macro_f1": float(rep["macro_avg"]["f1"]),
                "accuracy": float(accuracy_score_manual(y_val, y_pred)),
            })
        except MemoryError as e:
            # Manhattan bisa gagal untuk dataset besar — catat sebagai N/A
            results.append({
                "metric":   metric,
                "macro_f1": float("nan"),
                "accuracy": float("nan"),
                "error":    str(e),
            })
    return results