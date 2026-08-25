from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def cosine_tfidf(text1: str, text2: str) -> float:
    vec = TfidfVectorizer()
    X = vec.fit_transform([text1, text2])
    return float(cosine_similarity(X[0:1], X[1:2])[0, 0])


def build_vocab_vectors(text1: str, text2: str):
    vec = CountVectorizer()
    X = vec.fit_transform([text1, text2]).toarray()
    vocab = np.array(vec.get_feature_names_out())
    return X, vocab


def pearson_term_freq(text1: str, text2: str) -> float:
    X, _ = build_vocab_vectors(text1, text2)
    v1, v2 = X[0], X[1]

    if v1.sum() == 0 and v2.sum() == 0:
        return 0.0
    if np.std(v1) == 0 or np.std(v2) == 0:
        return 0.0

    return float(np.corrcoef(v1, v2)[0, 1])


def jaccard_terms(text1: str, text2: str) -> float:
    set1 = set(text1.split())
    set2 = set(text2.split())
    if not set1 and not set2:
        return 0.0
    inter = set1.intersection(set2)
    uni = set1.union(set2)
    return float(len(inter) / len(uni))
