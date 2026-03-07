# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# # 3. Propiedades estadísticas del lenguaje

# <a target="_blank" href="https://colab.research.google.com/github/umoqnier/cl-2026-2-lab/blob/main/notebooks/3_stats_properties.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# ## Objetivos

# - Mostrar el uso de CFG y derivados
#     - Ejemplos de parseo de dependencias
# - Ejemplificar etiquetado NER usando bibliotecas existentes
# - Explorar propiedades estadísticas del lenguaje natural y observar los siguientes fenomenos:
#     - La distribución de Zipf
#     - La distribución de Heap

# ## Perspectivas formales

# - Fueron el primer acercamiento al procesamiento del lenguaje natural. Sin embargo tienen varias **desventajas**
# - Requieren **conocimiento previo de la lengua**
# - Las herramientas son especificas de la lengua
# - Los fenomenos que se presentan son muy amplios y difícilmente se pueden abarcar con reglas formales (muchos casos especiales)
# - Las reglas tienden a ser rigidas y no admiten incertidumbre en el resultado

# ### Sintaxis

# ![](https://imgs.xkcd.com/comics/formal_languages_2x.png)
#
# **[audience looks around] 'What just happened?' 'There must be some context we're missing.'**

# #### Parsing basado en reglas

# - Gramaticas libres de contexto:
#
# $G = (T, N, O, R)$
# * $T$ símbolos terminales.
# * $N$ símbolos no terminales.
# * $O$ simbolo inicial o nodo raíz.
# * $R$ reglas de la forma $X \longrightarrow \gamma$ donde $X$ es no terminal y $\gamma$ es una secuencia de terminales y no terminales

plain_grammar = """
S -> NP VP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
PP -> P NP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
"""

# +
import nltk

grammar = nltk.CFG.fromstring(plain_grammar)
# Cambiar analizador y trace
analyzer = nltk.ChartParser(grammar)

sentence = "I shot an elephant in my pajamas".split()
trees = analyzer.parse(sentence)
# -

for tree in trees:
    print(tree, type(tree))
    print("\nBosquejo del árbol:\n")
    print(tree.pretty_print(unicodelines=True, nodedist=1))

# ## Perspectiva estadística

# - Puede integrar aspectos de la perspectiva formal
# - Lidia mejor con la incertidumbre y es menos rigida que la perspectiva formal
# - No requiere conocimiento profundo de la lengua. Se pueden obtener soluciones de forma no supervisada

# ## Modelos estadísticos

# - Las **frecuencias** juegan un papel fundamental para hacer una descripción acertada del lenguaje
# - Las frecuencias nos dan información de la **distribución de tokens**, de la cual podemos estimar probabilidades.
# - Existen **leyes empíricas del lenguaje** que nos indican como se comportan las lenguas a niveles estadísticos
# - A partir de estas leyes y otras reglas estadísticas podemos crear **modelos del lenguaje**; es decir, asignar probabilidades a las unidades lingüísticas

# ### Probabilistic Context Free Grammar

taco_grammar = nltk.PCFG.fromstring("""
O    -> FN FV     [0.7]
O    -> FV FN     [0.3]
FN   -> Sust      [0.6]
FN   -> Det Sust  [0.4]
FV   -> V FN      [0.8]
FV   -> FN V      [0.2]
Sust -> 'Juan'    [0.5]
Sust -> 'tacos'   [0.5]
Det  -> 'unos'    [1.0]
V    -> 'come'    [1.0]
""")
viterbi_parser = nltk.ViterbiParser(taco_grammar)

sentences = ["Juan come unos tacos", "unos tacos Juan come"]
for sent in sentences:
    for tree in viterbi_parser.parse(sent.split()):
        print(tree)
        print("Versión bosque")
        tree.pretty_print(unicodelines=True, nodedist=1)

# ### Parseo de dependencias

# Un parseo de dependencias devuelve las dependencias que se dan entre los tokens de una oración. Estas dependencias suelen darse entre pares de tokens. Esto es, que relaciones tienen las palabras con otras palabras.

# ##### Freeling - https://nlp.lsi.upc.edu/freeling/demo/demo.php

import spacy
from spacy import displacy

# !python -m spacy download es_core_news_sm

nlp = spacy.load("es_core_news_sm")

doc = nlp("La niña come un suani")

displacy.render(doc, style="dep")

for chunk in doc.noun_chunks:
    print("text::", chunk.text)
    print("root::", chunk.root.text)
    print("root dep::", chunk.root.dep_)
    print("root head::", chunk.root.head.text)
    print("=" * 10)

for token in doc:
    print("token::", token.text)
    print("dep::", token.dep_)
    print("head::", token.head.text)
    print("head POS::", token.head.pos_)
    print("CHILDS")
    print([child for child in token.children])
    print("=" * 10)

# #### Named Entity Recognition (NER)

# El etiquetado NER consiste en identificar "objetos de la vida real" como organizaciones, paises, personas, entre otras y asignarles su etiqueta correspondiente. Esta tarea es del tipo *sequence labeling* ya que dado un texto de entrada el modelo debe identificar los intervalos del texto y etiquetarlos adecuadamente con la entidad que le corresponde. Veremos un ejemplo a continuación.

# !pip install datasets

from datasets import load_dataset

# +
from huggingface_hub import login

login()
# -

# !hf auth whoami

data = load_dataset("MarcOrfilaCarreras/spanish-news")

# +
# data?

# +
import random

random.seed(42)
corpus = random.choices(data["train"]["text"], k=3)
docs = list(nlp.pipe(corpus))
for j, doc in enumerate(docs):
    print(f"DOC #{j + 1}")
    doc.user_data["title"] = " ".join(doc.text.split()[:10])
    for i, ent in enumerate(doc.ents):
        print(" -" * 10, f"Entity #{i}")
        print(f"\tTexto={ent.text}")
        print(f"\tstart/end={ent.start_char}-{ent.end_char}")
        print(f"\tLabel={ent.label_}")
# -

displacy.render(docs, style="ent")

# [Available labels](https://spacy.io/models/en)

# ## Leyes estadísticas

# Bibliotecas
from collections import Counter
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 6]

mini_corpus = """Humanismo es un concepto polisémico que se aplica tanto al estudio de las letras humanas, los
estudios clásicos y la filología grecorromana como a una genérica doctrina o actitud vital que
concibe de forma integrada los valores humanos. Por otro lado, también se denomina humanis-
mo al «sistema de creencias centrado en el principio de que las necesidades de la sensibilidad
y de la inteligencia humana pueden satisfacerse sin tener que aceptar la existencia de Dios
y la predicación de las religiones», lo que se aproxima al laicismo o a posturas secularistas.
Se aplica como denominación a distintas corrientes filosóficas, aunque de forma particular,
al humanismo renacentista1 (la corriente cultural europea desarrollada de forma paralela al
Renacimiento a partir de sus orígenes en la Italia del siglo XV), caracterizado a la vez por su
vocación filológica clásica y por su antropocentrismo frente al teocentrismo medieval
"""
words = mini_corpus.replace("\n", " ").split(" ")
len(words)

vocabulary = Counter(words)
vocabulary.most_common(10)

len(vocabulary)


# +
def get_frequencies(vocabulary: Counter, n: int) -> list:
    return [_[1] for _ in vocabulary.most_common(n)]


def plot_frequencies(frequencies: list, title="Freq of words", log_scale=False):
    x = list(range(1, len(frequencies) + 1))
    plt.plot(x, frequencies, "-v")
    plt.xlabel("Freq rank (r)")
    plt.ylabel("Freq (f)")
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    plt.title(title)


# -

frequencies = get_frequencies(vocabulary, 100)
plot_frequencies(frequencies)

plot_frequencies(frequencies, log_scale=True)

# **¿Qué pasará con más datos? 📊**

# ### Ley Zipf

# Utilizaremos otro corpus en español, entre más grande mejor (?)

dataset = load_dataset(
    "wikimedia/wikipedia", "20231101.es", split="train", streaming=True
)

# +
# dataset?
# -

row = next(iter(dataset))

print(row["title"])
print(row["text"])

corpus = dataset.take(10000)

# +
import re

def normalize_corpus(example):
    example["text"] = re.sub(r"[\W]", " ", example["text"])
    example["text"] = example["text"].lower()
    return example


# +
from datasets.iterable_dataset import IterableDataset


def count_words(corpus: IterableDataset) -> Counter:
    word_counts = Counter()
    normalized_corpus = corpus.map(normalize_corpus)
    for row in normalized_corpus:
        text = row["text"]
        word_counts.update(text.split())
    return word_counts


# -

# %%time
words = count_words(corpus)

words.most_common(10)

# +
import pandas as pd

def counter_to_pandas(counter: Counter) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(counter, orient="index").reset_index()
    df.columns = ["word", "count"]
    df.sort_values("count", ascending=False, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


# -

corpus_freqs = counter_to_pandas(words)

corpus_freqs.head(10)

corpus_freqs[corpus_freqs["word"] == "barriga"]

corpus_freqs["count"].plot(marker="o", legend=False)
plt.title("Ley de Zipf")
plt.xlabel("rank")
plt.ylabel("freq")
plt.show()

corpus_freqs["count"].plot(loglog=True, legend=False)
plt.title("Ley de Zipf (log-log)")
plt.xlabel("log rank")
plt.ylabel("log frecuencia")
plt.show()

# - Notamos que las frecuencias entre lenguas siguen un patrón
# - Pocas palabras (tipos) son muy frecuentes, mientras que la mayoría de palabras ocurren pocas veces
#
# De hecho, la frecuencia de la palabra que ocupa la posición r en el rank, es proporcional a $\frac{1}{r}$ (La palabra más frecuente ocurrirá aproximadamente el doble de veces que la segunda palabra más frecuente en el corpus y tres veces más que la tercer palabra más frecuente del corpus, etc)
#
# $$f(w_r) \propto \frac{1}{r^α}$$
#
# Donde:
# - $r$ es el rank que ocupa la palabra en el corpus
# - $f(w_r)$ es la frecuencia de la palabra en el corpus
# - $\alpha$ es un parámetro, el valor dependerá del corpus o fenómeno que estemos observando

# #### Formulación de la Ley de Zipf:

# #### ❓ ¿Cómo estimar el parámetro $\alpha$?

# Podemos hacer una regresión lineal minimizando la suma de los errores cuadráticos:
#
# $J_{MSE}=\sum_{r}^{}(log(f(w_{r}))-(log(c)-\alpha log(r)))^{2}$

# +
import numpy as np
from scipy.optimize import minimize

# Obtenemos los ranks y las frecuencias del corpus
# # +1 para hacer que los ranks inicien en 1 y no en 0
ranks = np.array(corpus_freqs.index) + 1  
frequencies = np.array(corpus_freqs["count"])

def zipf_minimization_objective(alpha: np.float64,
                               word_ranks: np.ndarray,
                               word_frequencies: np.ndarray) -> np.float64:
    """
    Calculate the sum of squared errors for Zipf's law fit.

    Parameters
    ----------
    alpha : np.float64
        The exponent parameter to optimize in Zipf's law
    word_ranks : np.ndarray
        Array of word ranks (1 = most frequent word)
    word_frequencies : np.ndarray
        Array of observed word frequencies

    Returns
    -------
    np.float64
        Sum of squared errors between log frequencies and Zipf's law prediction
    """
    predicted_log_freq = np.log(word_frequencies[0]) - alpha * np.log(word_ranks)
    return np.sum((np.log(word_frequencies) - predicted_log_freq) ** 2)

# Parámeto alfa inicial
initial_alpha_guess = 1.0

optimization_result = minimize(
    zipf_minimization_objective,
    initial_alpha_guess,
    args=(ranks, frequencies)
)
estimated_alpha = optimization_result.x[0]

mean_squared_error = zipf_minimization_objective(estimated_alpha, ranks, frequencies)
print(f"Estimated alpha: {estimated_alpha:.4f}\nMean Squared Error: {mean_squared_error:.4f}")


# -

def plot_generate_zipf(alpha: np.float64, ranks: np.ndarray, freqs: np.ndarray) -> None:
    plt.plot(
        np.log(ranks),
        np.log(freqs[0]) - alpha * np.log(ranks),
        color="r",
        label="Aproximación Zipf",
    )


plot_generate_zipf(estimated_alpha, ranks, frequencies)
plt.plot(np.log(ranks), np.log(frequencies), color="b", label="Distribución real")
plt.xlabel("log ranks")
plt.ylabel("log frecs")
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

# ### Ley de Heap

# Relación entre el número de **tokens** y **tipos** de un corpus
#
# $$T \propto N^b$$
#
# Dónde:
#
# - $T = $ número de tipos
# - $N = $ número de tokens
# - $b = $ parámetro  

# - **TOKENS**: Número total de palabras dentro del texto (incluidas repeticiones)
# - **TIPOS**: Número total de palabras únicas en el texto

# #### 📊 Ejercicio: Muestra el plot de tokens vs types
#
# - Hazlo para el corpus en español de wikipedia
# - Elige un corpus de wikipedia en otro idioma. Haz el mismo plot. ¿Qué observas?
#
# **HINT:** Obtener suma de tokens acumuladas` (`numpy.cumsum()`)

# +
# Tu código bonito acá 🔥
# -

# ## Diversidad lingüística 

# ### [Glottolog](https://glottolog.org/)

# Glottolog es uncatálogo de las lenguas, familias lingúísticas y dialectos del mundo (identificados como *languids*). Asigna a cada *languoid* un código único y estable. Los *languids* son organizados por clasificaciones genealógicas (un árbol de Glottolog) que está basado en archivos de investigaciones historicas comparables.
#
# Podemos [descargar los datos](https://glottolog.org/meta/downloads) de la plataforma gracias a su [licencia libre](https://creativecommons.org/licenses/by/4.0/). Para está práctica utilizarmos los archivo [`languages_and_dialects_geo.csv`](https://cdstar.eva.mpg.de//bitstreams/EAEA0-2198-D710-AA36-0/languages_and_dialects_geo.csv) y [`languoid.csv`](https://cdstar.eva.mpg.de//bitstreams/EAEA0-2198-D710-AA36-0/glottolog_languoid.csv.zip).

# +
import os

# Se asume que se han descargado los archivo y que se encuentra en la carpeta data/
DATA_PATH = "data"
LANG_GEO_FILE = "languages_and_dialects_geo.csv"
LANGUOID_FILE = "languoid.csv"
# -

languages = pd.read_csv(os.path.join(DATA_PATH, LANG_GEO_FILE))

languages.head()

languoids = pd.read_csv(os.path.join(DATA_PATH, LANGUOID_FILE))

languoids.head()

# +
# Mejorar estas coordenadas
min_lat = 14.5
max_lat = 32.7
min_long = -118.4
max_long = -86.8

mexico_languages = languages[
    (languages["latitude"] >= min_lat)
    & (languages["latitude"] <= max_lat)
    & (languages["longitude"] >= min_long)
    & (languages["longitude"] <= max_long)
]

# +
# Reconstrucción de linajes usando grafos locales (languoid.csv)
languoids_dict = languoids.set_index('id').to_dict('index')

def reconstruir_linaje(glottocode):
    """Sube por el árbol genealógico desde la lengua hasta la familia raíz."""
    linaje = []
    current_id = glottocode
    
    # Mientras el ID actual exista y no sea nulo (NaN)
    while pd.notna(current_id) and current_id in languoids_dict:
        nodo = languoids_dict[current_id]
        
        # Filtramos lenguas artificiales o "bookkeeping"
        if nodo.get('bookkeeping') or nodo.get('name') == 'Unclassifiable':
            return "Unclassifiable"
            
        # Insertamos el nombre al inicio de la lista para mantener el orden (Raíz -> Lengua)
        linaje.insert(0, str(nodo['name']))
        
        # Subimos al nodo padre
        current_id = nodo['parent_id']
        
    return " > ".join(linaje)


# +
# Aplicamos la función a nuestras lenguas de México
mexico_languages = mexico_languages.copy()
mexico_languages['tree'] = mexico_languages['glottocode'].apply(reconstruir_linaje)

# Filtramos las que no se pudieron clasificar
df = mexico_languages[~mexico_languages['tree'].isin(['', "Unclassifiable"])].copy()

# Extraemos la familia lingüística (la primera palabra del linaje)
df['Family'] = df['tree'].str.split().str[0]
df.set_index("glottocode", inplace=True)
df.head()
# -

p = df[df["name"] == "Huichol"]
p["tree"]["huic1243"]

# +
import plotly.graph_objects as go

def longest_common_prefix(str1, str2):
    """Calcula el ratio del prefijo común más largo entre dos linajes."""
    min_length = min(len(str1), len(str2))
    common_prefix = ""

    for i in range(min_length):
        if str1[i] == str2[i]:
            common_prefix += str1[i]
        else:
            break
            
    # Normalizamos el resultado
    return len(common_prefix) / min_length if min_length > 0 else 0

# Creamos la matriz de distancias vacía
n = len(df)
distance_matrix = pd.DataFrame(index=df.index, columns=df.index, dtype=float)

# Poblamos la matriz calculando la similitud por pares
for i in range(n):
    for j in range(i, n):
        distance = longest_common_prefix(df['tree'].iloc[i], df['tree'].iloc[j])
        distance_matrix.iloc[i, j] = distance
        distance_matrix.iloc[j, i] = distance

distance_df = pd.DataFrame(distance_matrix.values, index=df.index, columns=df.index)

# Ordenamos las lenguas por familia para una mejor visualización en el mapa de calor
ordered_languages = df.sort_values('Family').index
ordered_similarity_df = distance_df.reindex(index=ordered_languages, columns=ordered_languages)
# -

ordered_similarity_df.head()

# +
# Mapeamos los glottocodes a los nombres reales de las lenguas para las etiquetas
labels = ordered_similarity_df.columns.map(lambda x: df.loc[x, 'name'])

# Generamos el mapa de calor
fig = go.Figure(data=go.Heatmap(z=ordered_similarity_df, x=labels, y=labels))
fig.update_layout(title='Matriz de Similitud Genealógica',
                  xaxis={'side': 'top'},
                  width=1200, height=1200)
fig.show()
# -

# ## Práctica 2: Propiedades estadísticas del lenguaje y Diversidad
#
# ### Fecha de entrega: 17 de Marzo de 2026 11:59pm 
#
# **1. Verificación empírica de la Ley de Zipf**
#
# Verificar si la ley de Zipf se cumple en los siguientes casos:
#
# 1. En un lenguaje artificial creado por ustedes.
#     * Creen un script que genere un texto aleatorio seleccionando caracteres al azar de un alfabeto definido. 
#         * **Nota:** Asegúrense de incluir el carácter de "espacio" en su alfabeto para que el texto se divida en "palabras" de longitudes variables.
#     * Obtengan las frecuencias de las palabras generadas para este texto artificial
# 2. Alguna lengua de bajos recursos digitales (*low-resourced language*)
#     * Busca un corpus de libre acceso en alguna lengua de bajos recursos digitales
#     * Obten las frecuencias de sus palabras
#
# En ambos casos realiza lo siguiente:
# * Estima el parámetro $\alpha$ que mejor se ajuste a la curva
# * Generen las gráficas de rango vs. frecuencia (en escala y logarítmica).
#     * Incluye la recta aproximada por $\alpha$
# * ¿Se aproxima a la ley de Zipf? Justifiquen su respuesta comparándolo con el comportamiento del corpus visto en clase.
#
# > [!TIP]
# > Puedes utilizar los corpus del paquete [`py-elotl`](https://pypi.org/project/elotl/)
#
# ### 2. Visualizando la diversidad lingüística de México
#
# 1. Usando los datos de Glottolog filtralos con base en la región geográfica que corresponde a México
#     - Usa las columnas `"longitude"` y `"latitude"`
# 2. Realiza un plot de las lenguas por región de un mapa
#     - Utiliza un color por familia linguistica en el mapa
# 3. Haz lo mismo para otro país del mundo
#
# Responde las preguntas:
#
# - ¿Que tanta diversidad lingüística hay en México con respecto a otras regiones?
# - ¿Cuál es la zona que dirias que tiene mayor diversidad en México?
#
# > [!TIP]
# > Utiliza la biblioteca [`plotly`](https://plotly.com/python/getting-started/) para crear mapa interactivos
#
# ### EXTRA. Desempeño de NER en distintos dominios (Out-of-domain)
#
# Explorar la plataforma [Hugging Face Datasets](https://huggingface.co/datasets) y elegir documentos en Español provenientes de al menos 3 dominios muy distintos (ej. noticias, artículos médicos, tweets/redes sociales, foros legales).
# * Realizar Reconocimiento de Entidades Nombradas (NER) en muestras de cada dominio utilizando spaCy o la herramienta de su preferencia.
# * Mostrar una distribución de frecuencias de las etiquetas (PER, ORG, LOC, etc.) más comunes por dominio.
# * **Análisis:** Incluyan comentarios críticos sobre el desempeño observado. ¿En qué dominio el modelo cometió más errores y a qué creen que se deba estadísticamente?
#
# > [!TIP]
# > Utiliza bibliotecas con modelos preentrenados que te permitan realizar el etiquetado NER como [`spacy`](https://spacy.io/usage) o [`stanza`](https://stanfordnlp.github.io/stanza/#getting-started).
