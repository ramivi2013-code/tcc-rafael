# Comparador de Textos (TF-IDF + Embeddings)

Aplicação web minimalista para comparar dois documentos (`.txt`, `.pdf`, `.docx`) e medir interrelações
entre conteúdos.

O sistema calcula:
- **Similaridade Cosseno (TF-IDF)**
- **Correlação de Pearson** entre vetores de frequência de termos
- **Jaccard** de vocabulário
- **Similaridade Cosseno por Embeddings** (Sentence Transformers)
- **Top termos compartilhados** + **gráfico**
- **Resumo automático curto dos temas em comum** (heurístico interpretável)


## 1) Estrutura

```text
tcc-texto-compare/
  app.py
  text_pipeline.py
  readers.py
  metrics.py
  explain.py
  summarizer.py
  requirements.txt
  README.md
  .streamlit/
    config.toml
```

---

## 2) Pré-requisitos

- Python 3.9+
- Pip

---

## 3) Instalação

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

---

## 4) Executar

```bash
streamlit run app.py
```

---

## 5) Modelo de embedding padrão

Por padrão usamos:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Você pode trocar diretamente no `app.py` ou configurar via interface.

---

## 6) Observações

- Para PDFs digitalizados (imagem), será necessário OCR (futuro trabalho).
- O resumo automático implementado aqui é **heurístico e explicável**:
  ele usa termos compartilhados e frases mais representativas dos dois textos.

---

## 7) Ideias de extensão

- Comparação multi-documento (matriz de similaridades)
- OCR com `pytesseract`
- Detecção de tópicos (LDA/BERTopic)
- Exportação de relatório em PDF
