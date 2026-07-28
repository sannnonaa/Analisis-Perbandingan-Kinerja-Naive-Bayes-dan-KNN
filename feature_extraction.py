from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_vectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.70
):

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True   # PERBAIKAN BUG 13
    )

    return vectorizer


def fit_transform_tfidf(corpus, ngram_range=(1, 2), min_df=None, max_df=0.70):

    corpus_list = list(corpus)
    n = len(corpus_list)

    if min_df is None:
        if n < 10:
            min_df = 1
        elif n <= 30:
            min_df = 2
        else:
            min_df = 3

    vectorizer = build_tfidf_vectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )

    X = vectorizer.fit_transform(corpus_list)

    return X, vectorizer

# Transform data baru (testing / data uji / data baru)

def transform_tfidf(corpus, vectorizer):

    X = vectorizer.transform(corpus)
    return X

# Mengambil daftar fitur

def get_feature_names(vectorizer):
    try:
        return vectorizer.get_feature_names_out()
    except AttributeError:
        # get_feature_names_out() diperkenalkan di sklearn 1.0;
        # fallback ke get_feature_names() untuk versi lama.
        return vectorizer.get_feature_names()

# Ekstraksi detail TF-IDF per dokumen

def get_tfidf_detail(text, vectorizer):
    """
    Mengembalikan detail kalkulasi TF-IDF untuk satu dokumen teks
    menggunakan vectorizer yang sudah di-fit dari data latih.

    Digunakan untuk data latih MAUPUN data uji — keduanya
    menggunakan IDF yang sama (dari data latih) untuk mencegah
    data leakage.
    """
    import numpy as np

    vec_matrix   = vectorizer.transform([text])
    row_vec      = vec_matrix.toarray()[0]
    nz_indices   = np.where(row_vec > 0)[0]
    feature_names = vectorizer.get_feature_names_out()
    idf_vals     = vectorizer.idf_
    # Hitung N dari atribut vectorizer secara kompatibel lintas versi sklearn
    # n_samples_fit_ hanya ada di sklearn >= 1.1; fallback: hitung balik dari IDF
    # IDF = ln((N+1)/(df+1)) + 1  →  N = round(exp(IDF-1) * (df+1) - 1)
    # Gunakan term dengan df=1 (IDF tertinggi) untuk akurasi terbaik
    if hasattr(vectorizer, "n_samples_fit_"):
        n_docs = vectorizer.n_samples_fit_
    else:
        # Estimasi N dari nilai IDF maksimum (term paling langka, df=1)
        max_idf = float(np.max(idf_vals))
        n_docs  = int(round(np.exp(max_idf - 1) * (1 + 1) - 1))
    words        = text.split()

    result = []
    for f_idx in nz_indices:
        term       = feature_names[f_idx]
        term_words = term.split()

        # Hitung TF raw
        if len(term_words) == 1:
            tf_raw = words.count(term)
        else:
            tf_raw = sum(
                1 for i in range(len(words) - len(term_words) + 1)
                if words[i:i + len(term_words)] == term_words
            )

        idf_val = idf_vals[f_idx]

        # Hitung balik DF dari IDF: IDF = ln((1+N)/(1+df)) + 1
        df_val = round((1 + n_docs) / np.exp(idf_val - 1) - 1) if idf_val > 0 else 0

        result.append({
            "term"   : term,
            "tf_raw" : tf_raw,
            "tf_log" : round(1 + np.log(tf_raw), 4) if tf_raw > 0 else 0,
            "df"     : int(df_val),
            "idf"    : round(idf_val, 4),
            "tfidf"  : round(row_vec[f_idx], 4),
        })

    return sorted(result, key=lambda x: x["term"])


def get_tfidf_detail_batch(texts, vectorizer, source="train"):
    """
    Wrapper batch untuk get_tfidf_detail.
    Mengembalikan list of list of dict — satu list per dokumen.
    """
    results = []
    for text in texts:
        detail = get_tfidf_detail(text, vectorizer)
        for d in detail:
            d["source"] = source
        results.append(detail)
    return results