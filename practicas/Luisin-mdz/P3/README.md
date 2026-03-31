# Práctica 3: Representaciones vectoriales (BoW, TF‑IDF) + sesgos en embeddings

Este proyecto implementa (1) una comparación de **Bag of Words** vs **TF‑IDF** usando **similitud coseno** sobre un corpus pequeño en español, y (2) una exploración de **sesgos** en embeddings preentrenados (**GloVe**) mediante analogías.

---

## Contenido

```text
.
├── P3.py      # Código fuente
└── P3.ipynb   # Notebook generado
```

---

## Requisitos

Dependencias principales:

- `numpy`
- `pandas`
- `nltk`
- `scikit-learn`
- `gensim`
- `jupytext`

Ejemplo de instalación:

```bash
uv add numpy pandas nltk scikit-learn gensim jupytext
```

> Nota: el notebook descarga `punkt` de NLTK con `nltk.download("punkt")`.

---

## 1) BoW vs TF‑IDF (similitud coseno)

Flujo general:

- Preprocesamiento simple con `simple_preprocess()` (tokenización + filtros básicos)
- Construcción de matriz documento‑término con:
  - `CountVectorizer` (BoW)
  - `TfidfVectorizer` (TF‑IDF)
- Cálculo de similitud coseno entre una **query** y los documentos
- Comparación de scores en una tabla (`Score BOW` vs `Score TF‑IDF`)

---

## 2) Sesgos en embeddings (GloVe)

Se carga un embedding preentrenado:

- `glove-wiki-gigaword-100`

y se prueban analogías con `most_similar()` para observar asociaciones (p. ej. género ↔ profesiones / atributos).

También se discute una estrategia general de mitigación: **balancear el corpus** (ej., incluir formas masculinas y femeninas en proporciones similares).

---

## Uso de IA

Se utilizó **GitHub Copilot** como herramienta de apoyo en:

1. **Redacción de este README.**

---

## Autor

Luisin-mdz
