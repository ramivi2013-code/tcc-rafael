from __future__ import annotations

import re

# Stopwords PT via NLTK (fallback simples caso não tenha download)
try:
    from nltk.corpus import stopwords
    STOPWORDS_PT = set(stopwords.words("portuguese"))
except Exception:
    STOPWORDS_PT = {
        "a","o","os","as","de","do","da","dos","das","e","é","em","um","uma",
        "para","por","com","que","não","na","no","nos","nas","se","como","ao",
        "à","às","ou","mais","menos","muito","muita","muitos","muitas","ser",
        "ter","foi","era","são"
    }


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> str:
    """
    Pré-processamento simples e explicável:
    - lowercase
    - remove pontuação
    - mantém caracteres acentuados
    - remove stopwords PT
    - remove tokens curtos
    """
    text = text.lower()
    text = normalize_whitespace(text)
    # mantém letras, números e acentos
    text = re.sub(r"[^\wÀ-ÿ\s]", " ", text)
    text = normalize_whitespace(text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_PT and len(t) > 2]
    return " ".join(tokens)
