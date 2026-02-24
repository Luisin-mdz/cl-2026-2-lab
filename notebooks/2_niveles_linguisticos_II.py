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

# + [markdown] editable=true id="76b3a996-772a-4be8-a8eb-f1e9ae67d03e" slideshow={"slide_type": "slide"}
# # 2. Niveles Lingüísticos II

# + [markdown] editable=true slideshow={"slide_type": ""}
# <a target="_blank" href="https://colab.research.google.com/github/umoqnier/cl-2026-2-lab/blob/main/notebooks/2_niveles_linguisticos_II.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# + [markdown] editable=true id="615a09ab-2b52-440a-a4dc-fd8982c3c0e7" slideshow={"slide_type": "subslide"}
# ## Objetivos

# + [markdown] editable=true id="f034458c-9cb0-4966-a203-3145074c3fca" slideshow={"slide_type": ""}
# - Comparar enfoques basados en reglas y estadísticos para el análisis morfológico
# - Los alumnæs comprenderán la importancia de las etiquetas *POS* en tareas de _NLP_
# - Implementar un modelo de etiquetado automático
#     - Usando modelos discriminativos _HMMs_
#     - Usando modelos condicionales _CRFs_
#     - Contrastar ambos enfoques para generación automática de secuencias de etiquetas
# -

# # !uv add nltk scikit-learn sklearn-crfsuite <- Local con uv
# !pip install nltk scikit-learn sklearn-crfsuite

# + editable=true slideshow={"slide_type": "subslide"}
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import requests as r
from nltk.corpus import cess_esp

from rich import print as rprint

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF

# + [markdown] editable=true id="9fc31a40-1d6e-4c56-b07e-74a0c47a89c4" slideshow={"slide_type": "slide"}
# ## Morfología

# + [markdown] editable=true id="GJ10fzsXvFSS" slideshow={"slide_type": ""}
# <center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Flexi%C3%B3nGato-svg.svg/800px-Flexi%C3%B3nGato-svg.svg.png" width=200></center>
#
# > De <a href="//commons.wikimedia.org/wiki/User:KES47" class="mw-redirect" title="User:KES47">KES47</a> - <a href="//commons.wikimedia.org/wiki/File:Flexi%C3%B3nGato.png" title="File:FlexiónGato.png">File:FlexiónGato.png</a> y <a href="//commons.wikimedia.org/wiki/File:Nuvola_apps_package_toys_svg.svg" title="File:Nuvola apps package toys svg.svg">File:Nuvola apps package toys svg.svg</a>, <a href="http://www.gnu.org/licenses/lgpl.html" title="GNU Lesser General Public License">LGPL</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=27305101">Enlace</a>

# + [markdown] editable=true id="9677e8f1-aa3e-4f9c-8c7c-1422ea9ca913" slideshow={"slide_type": "subslide"}
# El análisis morfológico es la determinación de las partes que componen la palabra y su representación lingüística, es una especie de etiquetado
#
# Los elementos morfológicos son analizados para:
#
# - Determinar la función morfológica de las palabras
# - Hacer filtrado y pre-procesamiento de texto

# + [markdown] editable=true id="d962a952-6fa8-4410-82b6-8e7c6ba2f7e4" slideshow={"slide_type": "subslide"}
# ### Análisis morfológico basado en reglas

# + [markdown] editable=true id="cf70678d-f1d1-401e-aa23-da90a2ea7eaa" slideshow={"slide_type": "fragment"}
# Recordemos que podemos hacer un analizador morfológico haciendo uso de un transductor que vaya leyendo y haciendo transformaciones en una cadena. Formalmente:
#
# * $Q = \{q_0, \ldots, q_T\}$ conjunto finito de estados.
# * $\Sigma$ es un alfabeto de entrada.
# * $q_0 \in Q$ es el estado inicial.
# * $F \subseteq Q$ es el conjunto de estados finales.
#
# Un transductor es una 6-tupla $T = (Q, \Sigma, \Delta, q_0, F, \sigma)$ tal que
#
# * $\Delta$ es un alfabeto de salida teminal
# * $\Sigma$ es un alfabeto de entrada no terminal
# * $\sigma: Q \times \Sigma \times \Delta \longrightarrow Q$ función de transducción

# + [markdown] editable=true id="cdf136a8-6a9d-4434-b533-191553db242b" slideshow={"slide_type": "subslide"}
# #### Ejemplo: Parsing con expresiones regulares

# + [markdown] editable=true id="605d64c5-e102-4972-b5f6-92e0c08b495b" slideshow={"slide_type": "fragment"}
# Con fines de prácticidad vamos a _imitar_ el comportamiento de un transductor utilizando el modulo de python `re`

# + [markdown] editable=true id="cb409f23-1456-40bd-8fdd-997a862ad190" slideshow={"slide_type": "fragment"}
# La estructura del sustantivo en español es:
#
# ` BASE+AFIJOS (marcas flexivas)   --> Base+DIM+GEN+NUM`

# + editable=true id="1e2959df-96a4-40f5-a932-108abad269be" slideshow={"slide_type": "subslide"}
palabras = [
    "niño",
    "niños",
    "niñas",
    "niñitos",
    "gato",
    "gatos",
    "gatitos",
    "perritos",
    "paloma",
    "palomita",
    "palomas",
    "flores",
    "flor",
    "florecita",
    "lápiz",
    "lápices",
    #"chiquitititititos",
    #'curriculum', # curricula
    #'campus', # campi
]


# + editable=true id="66599e2e-d67b-49e2-9f80-0a20e756ca19" slideshow={"slide_type": "subslide"}
def morph_parser_rules(words: list[str]) -> list[str]:
    """Aplica reglas morfológicas a una lista de palabras para realizar
    un análisis morfológico.

    Parameters:
    ----------
    words : list of str
        Lista de palabras a las que se les aplicarán las reglas morfológicas.

    Returns:
    -------
    list of str
        Una lista de palabras después de aplicar las reglas morfológicas.
    """

    # Lista para guardar las palabras parseadas
    morph_parsing = []

    # Reglas que capturan ciertos morfemas
    # {ecita, itos, as, os}
    for w in words:
        # ecit -> DIM
        R0 = re.sub(r"([^ ]+)ecit([a|o|as|os])", r"\1-DIM\2", w)
        # it -> DIM
        R1 = re.sub(r"([^ ]+)it([a|o|as|os])", r"\1-DIM\2", R0)
        # a(s) -> FEM
        R2 = re.sub(r"([^ ]+)a(s)", r"\1-FEM\2", R1)
        # a -> FEM
        R3 = re.sub(r"([^ ]+)a\b", r"\1-FEM", R2)
        # o(s) -> MSC
        R4 = re.sub(r"([^ ]+)o(s)", r"\1-MSC\2", R3)
        # o .> MSC
        R5 = re.sub(r"([^ ]+)o\b", r"\1-MSC", R4)
        # es -> PL
        R6 = re.sub(r"([^ ]+)es\b", r"\1-PL", R5)
        # s -> PL
        R7 = re.sub(r"([^ ]+)s\b", r"\1-PL", R6)
        # Sustituye la c por z cuando es necesario
        parse = re.sub(r"c-", r"z-", R7)

        # Guarda los parseos
        morph_parsing.append(parse)
    return morph_parsing


# + editable=true slideshow={"slide_type": "subslide"}
def prettify_tags(word: str) -> str:
    tags = {
        "DIM": "[b bright_yellow]DIM[/]",
        "FEM": "[b green3]FEM[/]",
        "MSC": "[b medium_purple1]MSC[/]",
        "PL": "[b deep_sky_blue1]PL[/]",
    }
    for tag, pretty_tag in tags.items():
        word = word.replace(tag, pretty_tag)
    return word


# + colab={"base_uri": "https://localhost:8080/"} editable=true id="85c9648c-0c09-4d19-b52b-f020943caf5a" outputId="6804fc5f-edb7-423c-9bf3-96ec7fccc605" slideshow={"slide_type": "subslide"}
morph_parsing = morph_parser_rules(palabras)
for palabra, parseo in zip(palabras, morph_parsing):
    rprint(palabra, "-->", prettify_tags(parseo))

# + [markdown] editable=true id="3a104dae-e815-4271-9ff2-96802d50df9e" slideshow={"slide_type": "fragment"}
# #### Preguntas 🤔
# - ¿Qué pasa con las reglas en lenguas donde son más comunes los prefijos y no los sufijos?
# - ¿Cómo podríamos identificar características de las lenguas?

# + [markdown] editable=true id="8b3d6dac-6b02-45db-b352-a549a25fdabe" slideshow={"slide_type": "subslide"}
# #### Herramientas para hacer sistemas de análisis morfológico basados en reglas

# + [markdown] editable=true id="2ff2d85d-64c1-4272-9d6d-cf3599094588" slideshow={"slide_type": ""}
# - [Apertium](https://en.wikipedia.org/wiki/Apertium)
# - [Foma](https://github.com/mhulden/foma/tree/master)
# - [Helsinki Finite-State Technology](https://hfst.github.io/)
# - Ejemplo [proyecto](https://github.com/apertium/apertium-yua) de analizador morfológico de Maya Yucateco
# - Ejemplo normalizador ortográfico del [Náhuatl](https://github.com/ElotlMX/py-elotl/tree/master)
#
#
# También se pueden utilizar diferentes métodos de aprendizaje de máquina para realizar análisis/generación morfológica. En los últimos años ha habido un shared task de [morphological reinflection](https://github.com/sigmorphon/2023InflectionST) para poner a competir diferentes métodos

# + [markdown] editable=true id="d5b878ce-f60a-4069-b618-fcb0d4e77256" slideshow={"slide_type": "subslide"}
# ### Segmentación morfológica

# + [markdown] editable=true id="bcdc6126-3d69-4f90-9cfe-9dd6845525d5" slideshow={"slide_type": "fragment"}
# #### Corpus: [SIGMORPHON 2022 Shared Task on Morpheme Segmentation](https://github.com/sigmorphon/2022SegmentationST/tree/main)

# + [markdown] editable=true id="78102f5e-2bcc-41cd-981f-514d786cb9be" slideshow={"slide_type": "fragment"}
# - Shared task donde se buscaba convertir las palabras en una secuencia de morfemas
# - Dividido en dos partes:
#     - Segmentación a nivel de palabras ☀️
#     - Segmentación a nivel oraciones

# + [markdown] editable=true id="c6b0c352-e6ac-49db-9b3a-74da7db6ddad" slideshow={"slide_type": "subslide"}
# #### Track: words
#
# | word class | Description                      | English example (input ==> output)     |
# |------------|----------------------------------|----------------------------------------|
# | 100        | Inflection only                  | played ==> play @@ed                   |
# | 010        | Derivation only                  | player ==> play @@er                   |
# | 101        | Inflection and Compound          | wheelbands ==> wheel @@band @@s        |
# | 000        | Root words                       | progress ==> progress                  |
# | 011        | Derivation and Compound          | tankbuster ==> tank @@bust @@er        |
# | 110        | Inflection and Derivation        | urbanizes ==> urban @@ize @@s          |
# | 001        | Compound only                    | hotpot ==> hot @@pot                   |
# | 111        | Inflection, Derivation, Compound | trackworkers ==> track @@work @@er @@s

# + [markdown] editable=true id="84e2235d-f919-4c5d-832e-93551ba9ac4b" slideshow={"slide_type": "subslide"}
# #### Explorando el corpus

# + colab={"base_uri": "https://localhost:8080/", "height": 35} editable=true id="1a59cbf7-de9d-4618-8229-5cd369fa1c07" outputId="740f54b5-52b8-4665-863a-0969ec99fd86" slideshow={"slide_type": "fragment"}
response = r.get("https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/spa.word.test.gold.tsv")
# -



# + editable=true id="bb87e719-41c2-4e22-b9bf-5303902be165" slideshow={"slide_type": "subslide"}
LANGS = {
    "ces": "Czech",
    "eng": "English",
    "fra": "French",
    "hun": "Hungarian",
    "spa": "Spanish",
    "ita": "Italian",
    "lat": "Latin",
    "rus": "Russian",
}
CATEGORIES = {
    "100": "Inflection",
    "010": "Derivation",
    "101": "Inflection, Compound",
    "000": "Root",
    "011": "Derivation, Compound",
    "110": "Inflection, Derivation",
    "001": "Compound",
    "111": "Inflection, Derivation, Compound"
}


# + editable=true id="721ea2fa-24d8-4035-b0a5-87c86f821c6d" slideshow={"slide_type": "subslide"}
def get_track_files(lang: str, track: str = "word") -> list[str]:
    """Genera una lista de nombres de archivo del shared task

    Con base en el idioma y el track obtiene el nombre de los archivos
    para con información reelevante para hacer análisis estadístico.
    Esto es archivos .test y .dev

    Parameters:
    ----------
    lang : str
        Idioma para el cual se generarán los nombres de archivo.
    track : str, optional
        Track del shared task de donde vienen los datos (por defecto es "word").

    Returns:
    -------
    list[str]
        Una lista de nombres de archivo generados para el idioma y la pista especificados.
    """
    return [
        f"{lang}.{track}.test.gold",
        f"{lang}.{track}.dev",
    ]

# + editable=true slideshow={"slide_type": ""}



# + editable=true id="f583e168-1f5d-4426-9789-5fac8b2b221c" slideshow={"slide_type": "subslide"}
def get_raw_corpus(files: list) -> list:
    """Descarga y concatena los datos de los archivos tsv desde una URL base.

    Parameters:
    ----------
    files : list
        Lista de nombres de archivos (sin extensión) que se descargarán
        y concatenarán.

    Returns:
    -------
    list
        Una lista que contiene los contenidos descargados y concatenados
        de los archivos tsv.
    """
    result = []
    for file in files:
        print(f"Downloading {file}.tsv", end=" ")
        response = r.get(f"https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/{file}.tsv")
        print(f"status={response.status_code}")
        lines = response.text.split("\n")
        result.extend(lines[:-1])
    return result

# + editable=true slideshow={"slide_type": ""}



# + editable=true id="54de3b3d-be08-437d-b4ee-ff55d4fee2a9" slideshow={"slide_type": "subslide"}
def raw_corpus_to_dataframe(corpus_list: list, lang: str) -> pd.DataFrame:
    """Convierte una lista de datos de corpus en un DataFrame

    Parameters:
    ----------
    corpus_list : list
        Lista de líneas del corpus a convertir en DataFrame.
    lang : str
        Idioma al que pertenecen los datos del corpus.

    Returns:
    -------
    pd.DataFrame
        Un DataFrame de pandas que contiene los datos del corpus procesados.
    """
    data_list = []
    for line in corpus_list:
        try:
            word, tagged_data, category = line.split("\t")
        except ValueError:
            # Caso donde no existe la categoria
            word, tagged_data = line.split("\t")
            category = "NOT_FOUND"
        morphemes = tagged_data.split()
        data_list.append(
            {"words": word, "morph": morphemes, "category": category, "lang": lang}
        )
    df = pd.DataFrame(data_list)
    df["word_len"] = df["words"].apply(lambda x: len(x))
    df["morph_count"] = df["morph"].apply(lambda x: len(x))
    return df

# + editable=true slideshow={"slide_type": ""}



# + colab={"base_uri": "https://localhost:8080/"} editable=true id="fe645a77-f8ca-4bf1-b11e-2274b78ba6d1" outputId="d7e50464-31a2-4979-ca32-1ecdb7c07e5c" slideshow={"slide_type": "subslide"}
files = get_track_files("spa")
raw_spa = get_raw_corpus(files)
df = raw_corpus_to_dataframe(raw_spa, lang="spa")

# + colab={"base_uri": "https://localhost:8080/", "height": 206} editable=true id="774a3b39-3a37-43d5-afb4-5d98154afe9e" outputId="c99841e0-ea76-4343-9cca-0390ee68910c" slideshow={"slide_type": "fragment"}
df.head(20)

# + [markdown] editable=true id="0ffef737-bd71-43ed-be6d-a357819ab7c8" slideshow={"slide_type": "subslide"}
# #### Análisis cuantitativo para el Español

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="39965a92-4719-4043-919b-3b99dca0b8f9" outputId="bac4b6a4-efa9-40ce-ac86-51939ec0d6bd" slideshow={"slide_type": "fragment"}


# + editable=true id="c902ecf4-889a-4082-b4b8-0cb89ba9b16c" slideshow={"slide_type": "subslide"}
def plot_histogram(df, kind, lang):
    """Genera un histograma de frecuencia para una columna específica
    en un DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame que contiene los datos para generar el histograma.
    kind : str
        Nombre de la columna para la cual se generará el histograma.
    lang : str
        Idioma asociado a los datos.

    Returns:
    -------
    None
        Esta función muestra el histograma usando matplotlib.
    """
    counts = df[kind].value_counts().head(30)
    plt.bar(counts.index, counts.values)
    plt.xlabel(kind)
    plt.ylabel('Frequency')
    plt.title(f'{kind} Frequency Graph for {lang}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# + colab={"base_uri": "https://localhost:8080/", "height": 487} editable=true id="1f36d2ce-cca3-49f0-b989-69dae98f9646" outputId="b3648c42-99eb-452c-fb85-d73218829014" slideshow={"slide_type": "subslide"}
plot_histogram(df, "category", "spanish")


# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### Ejemplo 🥑: Adivina, adivinador, probabilidad del tipo de morfemas
#
# - No todos los afijos son iguales. Algunos sirven para conjugar (Flexivos) y otros para crear conceptos nuevos (Derivativos).
# - Aprovechando que el dataset de SIGMORPHON nos dice si una palabra es de tipo `Inflection` (100) o `Derivation` (010) calculemos qué tan probable es que un morfema sea flexivo o derivativo.
#

# + editable=true slideshow={"slide_type": "subslide"}
def analyze_morpheme_types(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula y obtiene las probabilidades de que un morfema sea flexivo o derivativo.
    """
    inflection_counts = defaultdict(int)
    derivation_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for _, row in df.iterrows():
        morphemes = row['morph'] 
        category = row['category']

        # Con base en SIGMORPHON:
        # Pos 0: Inflection, Pos 1: Derivation
        is_inflection = category[0] == '1'
        is_derivation = category[1] == '1'

        for morph in morphemes:
            clean_morph = morph.replace("@@", "")

            # Ignoramos morfemas muy cortos o vacíos
            if len(clean_morph) < 2:
                continue

            total_counts[clean_morph] += 1
            if is_inflection:
                inflection_counts[clean_morph] += 1
            if is_derivation:
                derivation_counts[clean_morph] += 1

    results = []
    for morph, total in total_counts.items():
        if total > 5:
            p_inflection = inflection_counts[morph] / total
            p_derivation = derivation_counts[morph] / total
            results.append({
                'morfema': morph,
                'total_freq': total,
                'prob_flexion': round(p_inflection, 3),
                'prob_derivacion': round(p_derivation, 3)
            })

    return pd.DataFrame(results).sort_values(by='total_freq', ascending=False)


# + editable=true slideshow={"slide_type": "subslide"}
morph_stats = analyze_morpheme_types(df)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### Los morfemas más "Gramaticales" (Flexivos)
#
# Deberíamos ver terminaciones verbales y plurales.

# + editable=true slideshow={"slide_type": "fragment"}
rprint("Top Morfemas Flexivos:")
rprint(morph_stats.sort_values(by=['prob_flexion', "total_freq"], ascending=False).head(10))

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### Los morfemas más "Léxicos" (Derivativos)
#
# Deberíamos ver sufijos que cambian sustantivos a adjetivos, etc.

# + editable=true slideshow={"slide_type": "fragment"}
rprint("Top Morfemas Derivativos:")
rprint(morph_stats.sort_values(by=['prob_derivacion', "total_freq"], ascending=False).head(10))

# + [markdown] editable=true id="408683e9-5d9b-44a6-bcb5-d2ad29bf7903" slideshow={"slide_type": "subslide"}
# #### Morfosintaxis

# + [markdown] editable=true id="10233e8c-b359-4ad7-ab6f-8d6d0c600a39" slideshow={"slide_type": "fragment"}
# - Etiquetas que hacen explícita la funcion gramatical de las palabras en una oración
# - Determina la función de la palabra dentro la oración (por ello se le llama Partes del Discurso)
# - Se le conoce tambien como **Análisis morfosintáctico**: es el puente entre la estructura de las palabras y la sintaxis
# - Permiten el desarrollo de herramientas de NLP más avanzadas
# - El etiquetado es una tarea que se puede abordar con técnicas secuenciales, por ejemplo, HMMs, CRFs, Redes neuronales

# + [markdown] editable=true id="8db57418-b7bc-4dc1-972d-aef484e9ea48" slideshow={"slide_type": ""}
# <center><img src="https://byteiota.com/wp-content/uploads/2021/01/POS-Tagging.jpg" height=500 width=500></center

# + [markdown] editable=true id="628dd2cd-c0b4-4b12-aa08-b74e5a81579c" slideshow={"slide_type": "subslide"}
# #### Ejemplo
#
# > El gato negro rie malvadamente
#
# - El - DET
# - gato - NOUN
# - negro - ADJ
# - ríe - VER
#
# <center><img src="https://i.pinimg.com/originals/0e/f1/30/0ef130b255ea704625b2ad473701dee5.gif"></center
# -

# ### Materia prima de otras tareas de NLP
#
# - Named entity recognition (NER)
# - Statistical language models
# - Text generation
# - Sentient analysis

# ## Etiquetado automático POS (*POS tagging*)
#
# - El etiquetado POS es una tarea del NLP dónde se le asigna de forma automática una etiqueta según su función a cada palabra tomando en cuenta el contexto de la oración.
#
# - En esta tarea se toma en cuenta cierta estructura de la oración.
#
# - En un enfoque probabilístico queremos obtener: $P(\overrightarrow{x},\overrightarrow{y})$
#
#
# donde:
# - $\overrightarrow{x}$ = $<x_1,x_2,...,x_n>$
# - $\overrightarrow{y}$ = $<y_1,y_2,...,y_n>$
# - $x_i = palabra$ y $y_i = etiqueta\ POS$

# ### Un primer acercamiento: Hidden Markov Models (HMM)

# <center><img src="https://www.davidsbatista.net/assets/images/2017-11-11-HMM.png"></center>

# $p(\overrightarrow{x},\overrightarrow{y}) = \displaystyle\prod_{i=1}^{n} p(y_i|y_{i-1}) ⋅ p(x_i|y_i)$
#
# Donde:
# - $\overrightarrow{y} = secuencia\ de\ etiquetas\ POS$
# - $\overrightarrow{x} = secuencia\ de\ palabras$

# ### Suposición de Markov
#
# > "The probability of a particular state is dependent only on the previous state"

# #### Características

# - Clasificador secuencial
#     - Dada una secuencia de entrada se predice la secuencia de salida más probable
#     - Se apreden los parámetros de secuencias previamente etiquetadas

# ### Componentes del los *HMM*

#
# - Estados (etiquetas): $T = t_1,t_2,...,t_n$
# - Observaciones (palabras): $W = w_1,w_2,...,w_n$
# - Estados iniciales y finales

# #### Probabilidades asociadas a estados

# - Matriz $A$ con las probabilidades de ir de un estado a otro
# - Matriz $B$ con las probabilidades de que una observasión se genera a partir de un estado
# - Probabilidades asociadas a los estados iniciales y finales

# ### ¿Qué soluciona HMM?

# 1. Aprender parámetros asociados con una secuencia de observación dada (training step)
#     - Dada una lista de palabras y sus etiquetas POS asociadas, el modelo aprende la estructura dada
# 2. Aplicar un modelo HMM previamente entrenado
#     - Dada una nueva oración nunca antes vista, se puede **predecir** la etiqueta POS asociada a cada palabra de dicha oración usando la estructura aprendida

# ### Corpus: `cess_esp`

# Corpus con 1M de palabras etiquetadas para Español y Catalán - https://www.nltk.org/book/ch02.html#tab-corpora
#

# Descargando el corpus cess_esp
nltk.download('cess_esp')

# Cargando oraciones
corpora = cess_esp.tagged_sents()

corpora[1][:5]


# +
def get_tags_map() -> dict:
    """sauce https://gist.github.com/vitojph/39c52c709a9aff2d1d24588aba7f8155/
    """
    tags_raw = r.get(
        "https://gist.githubusercontent.com/vitojph/39c52c709a9aff2d1d24588aba7f8155/raw/af2d83bc4c2a7e2e6dbb01bd0a10a23a3a21a551/universal_tagset-ES.map"
    ).text.split("\n")
    tags_map = {line.split("\t")[0].lower(): line.split("\t")[1] for line in tags_raw}
    return tags_map


def map_tag(tag: str, tags_map=get_tags_map()) -> str:
    if tags_map.get(tag.lower()) == ".":
        return "PUNCT"
    return tags_map.get(tag.lower(), "N/F")


def parse_tags(corpora: list[list[tuple]]) -> list[list[tuple]]:
    result = []
    for sentence in corpora:
        result.append([(word, map_tag(tag)) for word, tag in sentence])
    return result


# -

corpora = parse_tags(corpora)

corpora[0]

len(corpora)

# ### Implementación de HMMs

# Separando en dos conjuntos, uno para entrenamiento y otro para pruebas
train_data, test_data = train_test_split(corpora, test_size=0.3, random_state=42)

# Comprobemos la longitud de la data
len(train_data), len(test_data)

assert len(train_data) + len(test_data) == len(corpora), "Something is wrong with the split :("

# #### Entrenamiento

# +
from nltk.tag import hmm

# Creando el modelo HMM usando nltk
trainer = hmm.HiddenMarkovModelTrainer()

# Hora de entrenar
hmm_model = trainer.train(train_data)
# -

# #### Resultados

tagged_test_data = hmm_model.tag_sents([[word for word, _ in sent] for sent in test_data])

tagged_test_data[0]

# Extrayendo tags verdaderas vs tags predichas
y_true = [tag for sent in test_data for _, tag in sent]
y_pred = [tag for sent in tagged_test_data for _, tag in sent]

y_true[:3]

y_pred[:3]


def report_accuracy(y_true: list, y_pred: list) -> defaultdict:
    """Construye un reporte de exactitud por etiqueta."""
    label_accuracy_counts = defaultdict(lambda: {"correct": 0, "total": 0})

    for gold_tag, predicted_tag in zip(y_true, y_pred):
        label_accuracy_counts[gold_tag]["total"] += 1
        if gold_tag == predicted_tag:
            label_accuracy_counts[gold_tag]["correct"] += 1
    
    # Calculate and display the accuracy for each label
    print("Label\tAccuracy")
    for label, counts in label_accuracy_counts.items():
        accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        print(f"{label}\t{accuracy * 100:.2f}%")
    return label_accuracy_counts


label_accuracy_counts = report_accuracy(y_true, y_pred)

# ### Hablemos de Métricas

# #### Confusion Matrix (binaria)
#
# Es una forma tabular de vizualizar el desempeño de un modelo de *Machine Learning (ML)*. En las columnas tenemos la cuenta de etiquetas predichas mientras que en las filas tenemos la cuenta de las etiquetas reales (o viceversa)

# ![](https://i1.wp.com/dataaspirant.com/wp-content/uploads/2020/08/3_confusion_matrix.png?ssl=1)

# - **TP:** Etiquetas correctamente predichas como positivas.
#     - Ej: Se etiqueto un correo como spam y era spam
# - **FP:** Etiquetas incorrectamente predichas. 
#     - Ej: Se etiqueto un correo como spam y NO era spam
# - **TN:** Etiquetas correctamente predichas como negativas.
#     - Ej: Se etiqueto un correo como no spam y era no spam
# - **FN:** Etiquetas incorrectamente predichas como negativas.
#     - Ej: Se etiqueto un correo como no spam y era spam

# #### Accuracy = $\frac{TP + TN}{TP + TN + FP + FN}$
#
# Es una de las métricas más sencillas usadas en *ML*. Define que tan exacto es el modelo. Por ejemplo, si de 100 etiquetas el modelo acerto en 90 tendremos un accuracy de 0.90 o 90%

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_true))

# #### Precision = $\frac{TP}{TP + FP}$
#
# Indica la relación entre las predicciones positivas correctas (SPAM es SPAM) con el total de predicciones de la clase sin importar si fueron correctas o no (Todo lo que fue marcado como SPAM correctamente con todo lo que fue marcado como SPAM incorrectamente). *De los correos etiquetados como SPAM cuandos fueron efectivamente SPAM*

from sklearn.metrics import precision_score
print(precision_score(y_pred, y_true, average="macro"))

# #### Recall = $\frac{TP}{TP + FN}$
#
# Indica la relacion entre las predicciones positivas correctas con el total de predicciones incorrectas de otras clases (Todo lo que no se marco como SPAM cuando si era SPAM). *Todos los correos que en realidad eran SPAM*

from sklearn.metrics import recall_score
print(recall_score(y_pred, y_true, average="macro"))

# #### F1-score = $\frac{2PR}{P + R}$
#
# Es el promedio ponderado entre *precision* y *recall*. Toma en cuenta tanto los FP como los FN.

from sklearn.metrics import f1_score
print(f1_score(y_pred, y_true, average="macro"))

# #### Un ejemplo concreto

nltk.download('punkt_tab')

# +
unseen_sentence = "La casa es grande y luminosa."
#unseen_sentence = "La muchacha vio al dinosaurio con el telescopio"
#unseen_sentence = "El gato rie malvadamente"

# Tokenizando
tokenized_sentence = nltk.word_tokenize(unseen_sentence)

# Haciendo predicciones
predicted_tags = [tag for word, tag in hmm_model.tag(tokenized_sentence)]

print("Palabra \tPOS Tag (predicha)")
for word, tag in zip(tokenized_sentence, predicted_tags):
    print(f"{word}\t{tag}")
# -

# #### Comparando con modelos pre-entrenados

# - [Model information](https://spacy.io/models/es)

# !python -m spacy download es_core_news_sm

# +
import spacy
# spacy.cli.download("") # Direct from python

nlp = spacy.load("es_core_news_sm")
doc = nlp(unseen_sentence)
print([(w.text, w.pos_) for w in doc])
# -

from spacy import displacy
displacy.render(doc, style="dep", jupyter=True)

test_sentences = [" ".join([word for word, _ in sent]) for sent in test_data][:10]

docs = [nlp(sent) for sent in test_sentences]

for doc in docs[:1]:
    print()
    print([(w.text, w.pos_) for w in doc])

tagged_test_data[0]


# ### ¿Limitaciones?

# - Cada estado depende exclusivamente de su predecesor inmediato
#     - Dependencias limitadas
# - Cada observación depende exclusivamente del estado actual
# - Probabilidades estáticas
#     - Ejemplo, si veo un par de tags `(V) -> (S)` no importa si esta al inicio, en medio o al final de la oración la probabilidad siempre será la misma

# ![](https://3.bp.blogspot.com/-pPGGqs462yw/T1ol64kj9uI/AAAAAAAAAKI/CDCiH1IJodE/w1200-h630-p-k-nu/patricio.jpg)

# ## Sobrepasando las fronteras: _Conditional Random Fields (CRFs)_

# + [markdown] editable=true id="2276eafd-a3e4-48a8-b97a-359503a7d66f" slideshow={"slide_type": "fragment"}
# - Modelo de gráficas **no dirigido**. Generaliza los *HMM*
#     - Adiós a la *Markov assuption*
#     - Podemos tener cualquier dependencia que queramos entre nodos
#     - Nos enfocaremos en un tipo en concreto: *LinearChain-CRFs* ¡¿Por?!
#
# <center><img width=300 src="https://i.kym-cdn.com/entries/icons/original/000/032/676/Unlimited_Power_Banner.jpg"></center>
#

# + [markdown] editable=true id="c5a1bff5-1f06-416b-9244-c4eab4dd989a" slideshow={"slide_type": "subslide"}
# - Modela la probabilidad **condicional** $P(Y|X)$
#     - Modelo discriminativo
#     - Probabilidad de un estado oculto dada **toda** la secuecia de entrada
# ![homer](https://media.tenor.com/ul0qAKNUm2kAAAAd/hiding-meme.gif)

# + [markdown] editable=true id="74beab61-39ec-43bf-8ca8-44cbb9d62149" slideshow={"slide_type": "subslide"}
# - Captura mayor **número de dependencias** entre las palabras y captura más características
#     - Estas se definen en las *feature functions* 🙀
# - El entrenamiento se realiza aplicando gradiente decendente y optimización con algoritmos como [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
#
#
# <center><img src="https://iameo.github.io/images/gradient-descent-400.gif"></center>
#

# + [markdown] editable=true id="4653baf1-edc2-4813-be95-643a1b0f60f7" slideshow={"slide_type": "subslide"}
# $P(\overrightarrow{y}|\overrightarrow{x}) = \frac{1}{Z} \displaystyle\prod_{i=1}^N exp\{w^T ⋅ \phi(y_{i-1}, y_i, \overrightarrow{x}, i)\}$
#
# Donde:
# - $\overrightarrow{y}$ = Etiquetas POS
# - $\overrightarrow{x}$ = Palabras en una oración
# - $w^T$ = Vector de pesos a aprender
# - $\phi$ = Vector de *Features*
#     - Calculado con base en un conjunto de *feature functions*
# - $i$ = la posición actual en la oración
# - $Z$ = factor de normalización

# + [markdown] editable=true id="5c13a366-4bc3-425c-8db4-bb2986dc2e8f" slideshow={"slide_type": "subslide"}
# ![](https://aman.ai/primers/ai/assets/conditional-random-fields/Conditional_Random_Fields.png)
#
# Tomado de http://www.davidsbatista.net/blog/2017/11/13/Conditional_Random_Fields/

# + [markdown] editable=true id="5ad8c1dc-4c5b-41f8-95c2-40be89f8f07f" slideshow={"slide_type": "subslide"}
# #### Feature functions
#
# $\phi(y_{i-1}, y_i, \overrightarrow{x}, i)$
#
# - Parte fundamental de los CRFs
# - Cuatro argumentos:
#     - Todos los datos observables $\overrightarrow{x}$ (conectar $x$ con cualquier $y$)
#     - El estado oculto anterior $y_{i-1}$
#     - El estado oculto actual $y_i$
#     - El index del timestamp $i$
#         - Cada feature list puede tener diferentes formas

# + [markdown] editable=true id="c1cfd87f-d4d2-4501-bd47-5a7dc843db2b" slideshow={"slide_type": "subslide"}
# - Aquí es donde está la flexibilidad del modelo
# - Tantas *features* como queramos, las que consideremos que pueden ayudar a que el modelo tenga un mejor desempeño
#     - Íntimamente ligadas a la lengua. Para mejor desempeño se debe hacer un análisis de sus características.
# - Ejemplo:
#
# ```python
# [
#     "word.lower()",
#     "EOS",
#     "BOS",
#     "pre-word",
#     "nxt-word",
#     "word-position",
#     ...
# ]
# ```

# + [markdown] editable=true id="1c50c45f-9e37-42e0-9cfe-1a78429660ed" slideshow={"slide_type": "subslide"}
# ### Implementación de CRFs

# + [markdown] editable=true id="d7864024-2275-4cf6-93f4-16c6b0f54451" slideshow={"slide_type": "subslide"}
# #### Feature lists

# + editable=true id="2b6ab1f8-71fc-4862-ac2a-35fcb3ca5b2a" slideshow={"slide_type": ""}
def word_to_features(sent, i):
    word = sent[i][0]
    features = {
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
    }
    if i > 0:
        prev_word = sent[i - 1][0]
        features.update(
            {
                "prev_word.lower()": prev_word.lower(),
            }
        )
    else:
        # Beginning of sentence
        features["BOS"] = True
    return features


# Extract features and labels
def sent_to_features(sent) -> list:
    return [word_to_features(sent, i) for i in range(len(sent))]


def sent_to_labels(sent) -> list:
    return [label for token, label in sent]


# + colab={"base_uri": "https://localhost:8080/"} editable=true id="59da4b3c-7d1d-4725-936c-0afdbf74137f" outputId="8c9c0967-e2d2-4681-849b-dc4a9796bfff" slideshow={"slide_type": "subslide"}
len(corpora)
# -

corpora[0]

# + editable=true id="7266b709-0287-4f48-a62f-12ca9d7aaffb" slideshow={"slide_type": "fragment"}
# Preparando datos para el CRF
X = [sent_to_features(sent) for sent in corpora]
y = [sent_to_labels(sent) for sent in corpora]

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="f7f68f41-8ad8-42c4-b20d-626561047433" outputId="884befed-36dd-465f-ff62-ee8fee034187" slideshow={"slide_type": "fragment"}
# Exploración de data estructurada
print(X[0][0])
print(len(X[0]))
# -

print(y[0][0])
print(len(y[0]))

# + editable=true id="e6bdf707-c7fc-408b-9d81-65528e068cbb" slideshow={"slide_type": "subslide"}
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# + editable=true id="49503189-368e-4d83-abd2-b06433bae3e8" slideshow={"slide_type": "fragment"}
assert len(X_train) + len(X_test) == len(corpora), "Something wrong with my split :("
assert len(y_train) + len(y_test) == len(corpora), "Something wrong with my split :("
# -

# #### Entrenamiento

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="cba343a2-482f-4c67-a842-704ab5fc6f3e" outputId="3355fa0f-d2f3-4c66-de0b-595c8b754495" slideshow={"slide_type": "subslide"}
# Initialize and train the CRF tagger: https://sklearn-crfsuite.readthedocs.io/en/latest/api.html
crf = CRF(
    algorithm="lbfgs",
    c2=0.001,
    max_iterations=10,
    verbose=True,
)
try:
    crf.fit(X_train, y_train)
except AttributeError as e:
    print(e)
# -

# #### Evaluación

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="983f2a29-b455-4ca3-8115-eb4962e25481" outputId="d59aff27-781e-4b97-f262-7f39845c7e88" slideshow={"slide_type": "subslide"}
y_pred = crf.predict(X_test)

# Flatten the true and predicted labels
y_test_flat = [label for sent_labels in y_test for label in sent_labels]
y_pred_flat = [label for sent_labels in y_pred for label in sent_labels]

# Evaluate the model
report = classification_report(y_true=y_test_flat, y_pred=y_pred_flat)
rprint(report)
# -

# #### Ejercicio 🗺️: Experimentación y prueba del *CRF*
#
# - Experimentación
#     - Agrega más características a la función `word_to_features()`
#         - ¿Qué características pueden ser útiles?
#     - Experimenta con diferentes hiperparámetros del CRF
#     - En ambos casos observa cómo afectan su rendimiento
# - Prueba
#     - Usando el mejor modelo aprendido por el CRF etiqueta una oración desafiante
#     - Imprime el resultado
#     - ¿Las etiquetas son corretas? 



# + [markdown] editable=true id="1266180c-54ca-433c-bf77-e7052df67291" slideshow={"slide_type": "slide"}
# # Tarea 1: Exploración de Niveles del lenguaje 🔭

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# ### FECHA DE ENTREGA: 10 de Marzo 2026 at 11:59pm

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# ### Fonética
#
# 1. Con base en el sistema de búsqueda visto en la [práctica 1](https://github.com/umoqnier/cl-2026-2-lab/blob/main/notebooks/1_niveles_linguisticos_I.ipynb), dónde se recibe una palabra ortográfica y devuelve sus transcripciones fonológicas, proponga una solución para los casos en que la palabra buscada no se encuentra en el lexicón/diccionario.
#     - ¿Cómo devolver o **aproximar** su transcripción fonológica?
#     - Reutiliza el sistema de búsqueda visto en clase y mejóralo con esta funcionalidad.
#     - Muestra al menos tres ejemplos

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# ### Morfología
#
# 2. Elige tres lenguas del corpus que pertenezcan a familias lingüísticas distintas
#    - Ejemplo: `spa` (Romance), `eng` (Germánica), `hun` (Urálica)
#    - Para cada una de las tres lenguas calcula y compara:
#        -  **Ratio morfemas / palabra**: El promedio de morfemas que componen las palabras
#         -  **Indicé de Flexión / Derivación**: Del total de morfemas, ¿Qué porcentaje son etiquetas de flexión (`100`) y cuáles de derivación (`010`)?
# 3. Visualización
#     - Genera una figura con **subplots** para comparar las lenguas lado a lado.
#     - *Plot 1*: Distribución de la longitud de los morfemas
#     - *Plot 2*: Distribución de las categorías (flexión, derivación, raíz, etc.)
# 4. Con base en esta información, responde la pregunta: *¿Cuál de las tres lenguas se comporta más como una lengua aglutinante y cuál como una lengua aislante?*
#     - Justifica tu respuesta usando tus métricas y figuras

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# ### EXTRA:
#
# - Genera la [matriz de confusión](https://en.wikipedia.org/wiki/Confusion_matrix) para el etiquetador CRFs visto en clase
# - Observando las etiquetas donde el modelo falló responde las preguntas:
#     - ¿Por qué crees que se confundió?
#     - ¿Es un problema de ambigüedad léxica (la palabra tiene múltiples etiquetas)?
#     - ¿Qué *features* añadirías para solucionarlo?
