from __future__ import annotations

import streamlit as st
import pandas as pd

from readers import read_uploaded_file, SUPPORTED_EXTS
from text_pipeline import preprocess
from metrics import cosine_tfidf, pearson_term_freq, jaccard_terms
from explain import top_shared_terms
from summarizer import extract_common_theme_summary

# Embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# UI config
# ---------------------------
st.set_page_config(
    page_title="Comparador de Documentos",
    layout="wide",
    page_icon="📄"
)

# Minimal CSS for a clean, professional feel
st.markdown(
    """
    <style>
    .app-title {
        font-size: 2.0rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        font-size: 1.0rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 14px;
        background: rgba(255,255,255,0.6);
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="app-title">Comparador de Documentos</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'Compare dois arquivos (.txt, .pdf, .docx) usando TF-IDF, métricas estatísticas '
    'e embeddings semânticos.'
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------
# Sidebar settings
# ---------------------------
with st.sidebar:
    st.markdown("### Configurações")
    embedding_model_name = st.text_input(
        "Modelo de embedding",
        value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Modelo multilíngue leve e adequado para português."
    )
    top_n_terms = st.slider("Top termos compartilhados", 5, 30, 15)
    show_raw = st.checkbox("Mostrar texto original", value=False)

    st.markdown("---")
    st.markdown(
        "### Métricas exibidas\n"
        "- Similaridade Cosseno (TF-IDF)\n"
        "- Correlação de Pearson (freq. termos)\n"
        "- Jaccard (vocabulário)\n"
        "- Similaridade Cosseno (Embeddings)"
    )


# ---------------------------
# Upload area
# ---------------------------
c1, c2 = st.columns(2)

with c1:
    file1 = st.file_uploader(
        "Arquivo 1",
        type=list(SUPPORTED_EXTS),
        key="f1"
    )

with c2:
    file2 = st.file_uploader(
        "Arquivo 2",
        type=list(SUPPORTED_EXTS),
        key="f2"
    )


def compute_embedding_similarity(text1: str, text2: str, model_name: str) -> float:
    # Carrega modelo sob demanda
    model = SentenceTransformer(model_name)
    emb = model.encode([text1, text2], normalize_embeddings=True)
    sim = float(cosine_similarity([emb[0]], [emb[1]])[0, 0])
    return sim


if file1 and file2:
    try:
        r1 = read_uploaded_file(file1)
        r2 = read_uploaded_file(file2)
    except Exception as e:
        st.error(f"Erro ao ler arquivos: {e}")
        st.stop()

    raw1, raw2 = r1.text, r2.text

    # Pré-processamento p/ métricas baseadas em termos
    p1 = preprocess(raw1)
    p2 = preprocess(raw2)

    with st.spinner("Analisando documentos..."):
        # Métricas clássicas
        cosine_score = cosine_tfidf(p1, p2)
        pearson_score = pearson_term_freq(p1, p2)
        jaccard_score = jaccard_terms(p1, p2)

        # Explicabilidade
        shared = top_shared_terms(p1, p2, top_n=top_n_terms)
        shared_terms_only = [s.term for s in shared]

        # Embeddings (usa texto bruto para captar semântica)
        try:
            emb_score = compute_embedding_similarity(raw1, raw2, embedding_model_name)
        except Exception as e:
            emb_score = None
            st.warning(
                "Não foi possível calcular embedding. "
                "Verifique instalação de torch/sentence-transformers ou o nome do modelo."
            )

        # Resumo comum (heurístico)
        theme_summary = extract_common_theme_summary(raw1, raw2, shared_terms_only)

    # ---------------------------
    # Metrics row (styled)
    # ---------------------------
    st.markdown('<div class="section-title">Índices de Interrelação</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Cosseno (TF-IDF)", f"{cosine_score:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Pearson (freq.)", f"{pearson_score:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Jaccard", f"{jaccard_score:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with m4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if emb_score is None:
            st.metric("Cosseno (Embedding)", "—")
        else:
            st.metric("Cosseno (Embedding)", f"{emb_score:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        "Sugestão de leitura: TF-IDF captura sobreposição lexical; "
        "embeddings capturam proximidade semântica mesmo com vocabulário diferente."
    )

    # ---------------------------
    # Evidence
    # ---------------------------
    st.markdown('<div class="section-title">Evidências e Explicabilidade</div>', unsafe_allow_html=True)

    if shared:
        df = pd.DataFrame([{
            "termo": s.term,
            "compartilhado": s.shared_score,
            "freq_texto1": s.freq_texto1,
            "freq_texto2": s.freq_texto2,
        } for s in shared])

        left, right = st.columns([1, 1])

        with left:
            st.write("Top termos compartilhados")
            st.dataframe(df, use_container_width=True, hide_index=True)

        with right:
            st.write("Gráfico dos termos mais compartilhados")
            chart_df = df.set_index("termo")[["compartilhado"]]
            st.bar_chart(chart_df)

    else:
        st.info("Não foram encontrados termos compartilhados relevantes após o pré-processamento.")

    # ---------------------------
    # Theme summary
    # ---------------------------
    st.markdown('<div class="section-title">Resumo curto dos temas em comum</div>', unsafe_allow_html=True)
    st.write(theme_summary)

    # ---------------------------
    # Raw text (optional)
    # ---------------------------
    if show_raw:
        with st.expander("Visualizar textos originais"):
            a, b = st.columns(2)
            with a:
                st.caption(f"{r1.filename}")
                st.text_area("Texto 1", raw1, height=260)
            with b:
                st.caption(f"{r2.filename}")
                st.text_area("Texto 2", raw2, height=260)

else:
    st.info("Envie dois arquivos para iniciar a comparação.")
