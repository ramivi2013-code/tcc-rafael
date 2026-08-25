from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List

from metrics import build_vocab_vectors


@dataclass
class SharedTerm:
    term: str
    shared_score: int
    freq_texto1: int
    freq_texto2: int


def top_shared_terms(text1: str, text2: str, top_n: int = 20) -> List[SharedTerm]:
    X, vocab = build_vocab_vectors(text1, text2)
    v1, v2 = X[0], X[1]
    shared = np.minimum(v1, v2)

    idx = np.argsort(shared)[::-1]
    out: List[SharedTerm] = []
    for i in idx:
        if shared[i] <= 0:
            break
        out.append(
            SharedTerm(
                term=str(vocab[i]),
                shared_score=int(shared[i]),
                freq_texto1=int(v1[i]),
                freq_texto2=int(v2[i]),
            )
        )
        if len(out) >= top_n:
            break
    return out
