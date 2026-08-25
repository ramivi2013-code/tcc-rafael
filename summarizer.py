from __future__ import annotations

import re
from typing import List, Tuple


def split_sentences(text: str) -> List[str]:
    # Segmentação simples e robusta para PT
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # separa por pontuação forte
    sents = re.split(r"(?<=[\.!?])\s+", text)
    # limpa sentenças muito curtas
    sents = [s.strip() for s in sents if len(s.strip()) >= 30]
    return sents


def extract_common_theme_summary(
    raw_text1: str,
    raw_text2: str,
    shared_terms: List[str],
    max_sentences: int = 3
) -> str:
    """
    Resumo curto e explicável dos temas em comum.

    Heurística:
    - pega termos mais compartilhados (já filtrados)
    - procura sentenças de cada texto que contenham esses termos
    - seleciona as sentenças mais "densas" em termos compartilhados
    """
    if not shared_terms:
        return "Os textos compartilham poucos termos relevantes após o pré-processamento."

    s1 = split_sentences(raw_text1)
    s2 = split_sentences(raw_text2)

    def score_sentence(sent: str) -> int:
        low = sent.lower()
        return sum(1 for t in shared_terms[:10] if t in low)

    scored = []
    for s in s1:
        sc = score_sentence(s)
        if sc > 0:
            scored.append((sc, s))
    for s in s2:
        sc = score_sentence(s)
        if sc > 0:
            scored.append((sc, s))

    if not scored:
        # fallback explicável
        terms = ", ".join(shared_terms[:8])
        return f"Temas em comum sugeridos pelos termos mais frequentes: {terms}."

    scored.sort(key=lambda x: x[0], reverse=True)

    chosen = []
    seen = set()
    for sc, s in scored:
        key = s[:80]
        if key in seen:
            continue
        seen.add(key)
        chosen.append(s)
        if len(chosen) >= max_sentences:
            break

    # Se ainda ficou muito curto, complementa com lista de termos
    terms = ", ".join(shared_terms[:8])
    text = " ".join(chosen)
    return f"{text} Termos centrais compartilhados: {terms}."
