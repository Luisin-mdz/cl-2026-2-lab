# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Práctica 3 Representaciones Vectoriales

# %% [markdown]
# ## Reutilizamos las funciones vistas en el notebook de clase.

# %%
import re
import pandas as pd 
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
# %%
def simple_preprocess(text: str):
    tokens = word_tokenize(text.lower(), language='spanish')
    return [word for word in tokens if word.isalnum() and len(word) > 1 and not re.match(r'^\d+$', word)]
# %%
def create_bow_dataframe(docs_raw: list, titles: list[str], vectorizer) -> pd.DataFrame:
    # fit_transform ajusta el vocabulario y crea la matriz en un solo paso
    matrix = vectorizer.fit_transform(docs_raw)

    # Podemos crear el DataFrame directamente pasando la matriz a un array tradicional
    # vectorizer.get_feature_names_out() nos da la lista de palabras en el orden exacto de las columnas
    df = pd.DataFrame(
        matrix.toarray(), index=titles, columns=vectorizer.get_feature_names_out()
    )
    return df
# %%
doc1 = """Resolver el ajedrez por computadora, sin embargo, plantea un problema que no podrá
solucionarse incluso en el futuro: el crecimiento exponencial de las posibles jugadas y
variantes que pueden producirse."""
doc2 = """Deep Thought es el antecesor de la famosa máquina de ajedrez de IBM, Deep Blue 202.
Todo inició como un proyecto en la Universidad Carnegie Mellon, cuando un estudiante
graduado, Feng-hsiung Hsu, comenzó a trabajar en su proyecto de tesis: una máquina que
juega al ajedrez a la que él llamó ChipTest. """
doc3 = """ La verdad es que la teoría hipermoderna no es otra cosa que la aplicación, durante el desarrollo
de la apertura, de los mismos viejos principios, pero poniendo en práctica tácticas un tanto
novedosas."""
doc4 = """ Pensaba en ti, Susana. En las lomas verdes. Cuando volábamos papalotes[57] en la época del aire. Oíamos allá abajo el rumor viviente del pueblo mientras estábamos encima de él, arriba de la loma, en tanto se nos iba el hilo de cáñamo arrastrado por el viento. “Ayúdame, Susana.” Y unas manos suaves se apretaban a nuestras manos. “Suelta más hilo.”
»El aire nos hacía reír; juntaba la mirada de nuestros ojos, mientras el hilo corría entre los dedos detrás del viento, hasta que se rompía con un leve crujido como si hubiera sido trozado por las alas de algún pájaro. Y allá arriba, el pájaro de papel caía en maromas[58] arrastrando su cola de hilacho, perdiéndose en el verdor de la tierra.
»Tus labios estaban mojados como si los hubiera besado el rocío.»
"""
doc5 = """¿Por qué aquella mirada se volvía valiente ante la resignación? Qué le costaba a él perdonar, cuando era tan fácil decir una palabra o dos, o cien palabras si éstas fueran necesarias para salvar el alma. ¿Qué sabía él del Cielo y del Infierno?
Y sin embargo, él, perdido en un pueblo sin nombre, sabía los que habían merecido el Cielo. Había un catálogo. Comenzó a recorrer los santos del panteón católico comenzando por los del día: «Santa Nunilona, virgen y mártir; Anercio, obispo; Santas Salomé viuda, Alodia o Elodia y Nulina, vírgenes; Córdula y Donato.» Y siguió. Ya iba siendo dominado por el sueño cuando se sentó en la cama: «Estoy repasando una hilera de santos como si estuviera viendo saltar cabras.»
Salió fuera y miró el cielo. Llovían estrellas. Lamentó aquello porque hubiera querido ver un cielo quieto. Oyó el canto de los gallos. Sintió la envoltura de la noche cubriendo la tierra. La tierra, «este valle de lágrimas»."""
# El ratio dorado: 5 repeticiones para engañar a BoW y 16 rarezas para hackear TF-IDF
doc6 = "ajedrez ajedrez ajedrez ajedrez ajedrez lomas verdes papalotes viviente cáñamo suaves apretaban crujido trozado maromas hilacho rocío mojados besado verdor labios"
# %%
docs_raw = [doc1, doc2, doc3, doc4, doc5, doc6]
# %%
vectorizer = CountVectorizer(tokenizer=simple_preprocess,token_pattern=None)
bag_of_words_corpus = vectorizer.fit_transform(docs_raw)
diccionario = vectorizer.vocabulary_
sorted(diccionario.items(), key=lambda item: item[1])

# %%
for column_idx, word in enumerate(vectorizer.get_feature_names_out()):
    print(column_idx,word)

# %%
bag_of_words_corpus.toarray()

# %%
print(len(bag_of_words_corpus.toarray()))
len(bag_of_words_corpus.toarray()[1])

# %%
titulos = ["ajedrez por compu","DeepBlue","Fundamentos","Pedro Páramo1","Pedro Páramo2","query_tramposo"]
docs_matrix = create_bow_dataframe(
    docs_raw,
    titulos,
    vectorizer = CountVectorizer(tokenizer = simple_preprocess, token_pattern = None)
)

# %%
docs_matrix

# %%
# 1. Extraemos el vector de la query y le damos la forma (1, -1) que pide sklearn
vector_query = docs_matrix.loc["query_tramposo"].values.reshape(1, -1)

# 2. Creamos una lista vacía para guardar los resultados y armar la tabla después
resultados_bow = []

# 3. Hacemos el for loop iterando sobre los 5 primeros títulos (sin incluir la query)
for titulo in titulos[:5]: 
    # Extraemos el vector del documento actual
    vector_doc = docs_matrix.loc[titulo].values.reshape(1, -1)
    
    # Calculamos la similitud (nos devuelve una matriz de 1x1, por eso el [0][0])
    similitud = cosine_similarity(vector_query, vector_doc)[0][0]
    
    # Guardamos el resultado
    resultados_bow.append((titulo, similitud))
    
    # Lo imprimimos para ir viéndolo
    print(f"Similitud Query vs {titulo}: {similitud:.4f}")

# %% [markdown]
# ### Ahora en TF-IDF

# %%
doc_matrix_tfidf = create_bow_dataframe(
    docs_raw,
    titulos,
    vectorizer = TfidfVectorizer(tokenizer = simple_preprocess, token_pattern = None)
)

# %%
vector_query_tfidf = doc_matrix_tfidf.loc["query_tramposo"].values.reshape(1,-1)
resultados_tfidf=[]
for titulo in titulos[:5]:
    vector_doc_tfidf = doc_matrix_tfidf.loc[titulo].values.reshape(1,-1)
    similitud_tfidf = cosine_similarity(vector_query_tfidf, vector_doc_tfidf)[0][0]
    resultados_tfidf.append((titulo,similitud_tfidf))
    print(f"similitud TD-IDF query vs {titulo}:{similitud_tfidf:.4f}")

# %%
resultados_tfidf

# %%
scores_bow = [resultado[1] for resultado in resultados_bow]
scores_tfidf = [resultado[1] for resultado in resultados_tfidf]
nombre_docs = titulos[:5]
df_comp = pd.DataFrame({
    "Score BOW": scores_bow,
    "Score TF-IDF": scores_tfidf
}, index = nombre_docs)

# %%
df_comp

# %% [markdown]
# #### Hubo un cambio del Score BoW a TF-IDF
# Podemos notar que con la query tramposa en BoW la frecuencia de ‘ajedrez’ domina por lo que en el documento 3 Deep Blue comparte mayor similitud que en las demás, pero al tener más palabras únicas en comun con Pedro Páramo1, al hacer TF-IDF, dicho documento tiene muchas mayores coicidencias con la query tramposa.

# %% [markdown]
# ## Parte 2: Busqueda de sesgos.

# %%
import gensim.downloader as gensim_api
word_vectors = gensim_api.load("glove-wiki-gigaword-100")

# %%
print(word_vectors.most_similar(positive=['man', 'profession'], negative=['woman']))
print()
print(word_vectors.most_similar(positive=['woman', 'profession'], negative=['man']))

# %% [markdown]
# ### Analizando los sesgos genéricos relacionados a las profesiones de hombre y mujeres
# Podemos notar que las profesiones más ligadas a las mujeres son todas aquellas que se dedican a cuidar, mientras que en los hombres son palabras enfocadas al conocimiento

# %%
print(word_vectors.most_similar(positive=['man', 'brilliant'], negative=['woman']))
print()
print(word_vectors.most_similar(positive=['woman', 'brilliant'], negative=['man']))

# %% [markdown]
# ### Analizando la palabra "Brilliant"
# Notemos que con "brilliant" los atributos masculinos están todos relacionados al intelecto y a las ciencias, en cambio los atributos femeninos están todos relacionados con la apariencia

# %% [markdown]
# ## Una forma de mitigar los sesgos de género al entrenar Modelos de lenguaje
# Una forma de mitigar estos sesgos podría ser balancear la cantidad de atributos relacionados con los hombres y las mujeres. Por ejemplo, si la oración original menciona a "los ingenieros", el texto puede ser modificado o replicado para mencionar a "las ingenieras", logrando así que al final el corpus de entrenamiento tenga la misma cantidad de oraciones asociadas a un género u otro.
