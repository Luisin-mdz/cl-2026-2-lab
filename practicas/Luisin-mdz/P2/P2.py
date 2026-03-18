#!/usr/bin/env python3
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
# # P2: Ley de Zipf
# Veremos si un idioma artificial creado aleatoreamente cumple con la ley de Pzif, y veremos como se comporta una lengua con pocos recursos digitales.
# %%
# Importamos las librerías necesarias
import pdfplumber
import re
import numpy as np
import pandas as pd
from collections import Counter
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
# %% [markdown]
# ## Procesamiento de texto
# %%
def limpia_pdf(pdf_path):
    texto = ""
    with pdfplumber.open(pdf_path) as pdf:
        for pagina in pdf.pages:
            texto += pagina.extract_text() + "\n"
    texto = texto.lower()
    texto = re.sub(r"\d+", "", texto)
    texto = re.sub(r"[^\w\s']", " ", texto)
    return texto.split()

def contar_palabras(texto):
    conteo = Counter(texto)
    return conteo

def crear_dataframe(conteo):
    df = pd.DataFrame(conteo.items(), columns = ["Palabras","Frecuencia"])
    df = df.sort_values("Frecuencia", ascending = False,).copy()
    df["Rango"] = range(1,len(df)+1)
    return df
# %% [markdown]
# ## Generando un idioma artificial.
# %%
# Definimos el alfabeto y la longitud de las palabras (con un espacio)
alfabeto = "abcdefghijklmnñopqrstuvwxyz "
letras_sueltas = np.random.choice(list(alfabeto), size=5000000, replace=True)
chorizo_de_letras = "".join(letras_sueltas)
texto_artificial = chorizo_de_letras.split()
# %%
# Longitud de nuestro texto artificial, 
len(texto_artificial)
# %% [markdown]
# ## procesamos el texto artificial y analizamos su distribución de palabras.
# %%
conteo_artificial = contar_palabras(texto_artificial)
df = crear_dataframe(conteo_artificial)
df.head(15)
# %%
plt.plot(df["Rango"],df["Frecuencia"],marker = "o")
plt.title("Ley de Zipf")
plt.xlabel("Rango")
plt.ylabel("Frecuencia")
plt.show()
# %%
plt.loglog(df["Rango"], df["Frecuencia"], marker="o")
plt.title("Ley de Zipf(Log-Log)")
plt.xlabel("Rango")
plt.ylabel("Frecuencia")
plt.show()
# %% [markdown]
# ## Estimando el parametro alfa (usando el notebook de clase)
# %%
ranks = np.array(df["Rango"])
frequencies = np.array(df["Frecuencia"])
# %%
def zipf_minimization_objective(
    alpha: np.float64, word_ranks: np.ndarray, word_frequencies: np.ndarray
) -> np.float64:
    predicted_log_freq = np.log(word_frequencies[0]) -alpha * np.log(word_ranks)
    return np.sum((np.log(word_frequencies) - predicted_log_freq) ** 2)
# %%
initial_alpha_guess = 1.0

optimization_result = minimize(
    zipf_minimization_objective,
    initial_alpha_guess,
    args=(ranks, frequencies)
)

estimated_alpha = optimization_result.x[0]

mean_squared_error = zipf_minimization_objective(
    estimated_alpha, ranks, frequencies
)

print(f"Estimated alpha: {estimated_alpha:.4f}")
print(f"Mean Squared Error: {mean_squared_error:.4f}")
# %%
plt.loglog(ranks, frequencies, "o", label="Datos")

zipf_fit = frequencies[0] * ranks**(-estimated_alpha)

plt.loglog(ranks, zipf_fit, label=f"Zipf α={estimated_alpha:.2f}")

plt.title("Ley de Zipf (log - log)")
plt.xlabel("Rango")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()
# %% [markdown]
# ## ¿Se aproxima a la ley de Zipf?
# No, podemos ver que la curva no se ajusta a los datos (log-log) y que el parametro alfa no está cerca de ser 1 como pasa en los lenguajes naturales, asi que nuestro texto aleatorio no sigue la ley de Zipf

# %% [markdown]
# ## Analizando un corpus con pocos recursos digitales
# Analizaremos una biblia de Tzotzil de los altos de chiapas, exáctamente del Tzotzil de Zinacantán.
# %%
biblia_tzotzil = limpia_pdf("biblia_tzotzil.pdf")
conteo_tzo = contar_palabras(biblia_tzotzil)
df_tzo = crear_dataframe(conteo_tzo)
# %%
# DataFrame ordenado 
df_tzo.head(20)

# %%
plt.plot(df_tzo["Rango"],df_tzo["Frecuencia"],marker = "o")
plt.title("Ley de Zipf")
plt.xlabel("Rango")
plt.ylabel("Frecuencia")
plt.show()

# %%
plt.loglog(df_tzo["Rango"],df_tzo["Frecuencia"],marker = "o")
plt.title("Ley de Zipf (log-log)")
plt.xlabel("Rango")
plt.ylabel("Frecuencia")
plt.show()

# %%
ranks = np.array(df_tzo["Rango"])
frequencies = np.array(df_tzo["Frecuencia"])
initial_alpha_guess = 1.0

optimization_result = minimize(
    zipf_minimization_objective,
    initial_alpha_guess,
    args=(ranks, frequencies)
)

estimated_alpha = optimization_result.x[0]

mean_squared_error = zipf_minimization_objective(
    estimated_alpha, ranks, frequencies
)

print(f"Estimated alpha: {estimated_alpha:.4f}")
print(f"Mean Squared Error: {mean_squared_error:.4f}")

# %%
plt.loglog(ranks, frequencies, "o", label="Datos")

zipf_fit = frequencies[0] * ranks**(-estimated_alpha)

plt.loglog(ranks, zipf_fit, label=f"Zipf α={estimated_alpha:.2f}")

plt.title("Ley de Zipf (log - log)")
plt.xlabel("Rango")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()

# %% [markdown]
# ### ¿Se aproxima a la ley de Zipf?
# Si, notemos que la gŕafica logaritmica se acerca mucho a la recta generada por el alfa calculada anteriormente, al ser el Tzotzil una lengua natural claramente sigue la ley de Zipf.

# %% [markdown]
# ## 2. Visualizando la diversidad linguistica de México y Chile
# Reutilizaremos el código visto en clase.

# %%
DATA_PATH = "data"
LANG_GEO_FILE = "languages_and_dialects_geo.csv"
LANGUOID_FILE = "languoid.csv"

# %%
languages = pd.read_csv(os.path.join(DATA_PATH, LANG_GEO_FILE))
languoids = pd.read_csv(os.path.join(DATA_PATH, LANGUOID_FILE))

# %%
languages.head(15)
languoids.head(15)

# %%
min_lat = -14
max_lat = 33
min_long = -118
max_long = -86

mexico_languages = languages[
    (languages["latitude"] >= min_lat)
    &(languages["latitude"] <= max_lat)
    &(languages["longitude"] >= min_long)
    &(languages["longitude"] <= max_long)
]

# %%
len(mexico_languages)

# %% [markdown]
# ### Tomamos la funcion de reconstruir linajes de la práctica

# %%
# Reconstrucción de linajes usando grafos locales (languoid.csv)
languoids_dict = languoids.set_index("id").to_dict("index")


def reconstruir_linaje(glottocode):
    """Sube por el árbol genealógico desde la lengua hasta la familia raíz."""
    linaje = []
    current_id = glottocode

    # Mientras el ID actual exista y no sea nulo (NaN)
    while pd.notna(current_id) and current_id in languoids_dict:
        nodo = languoids_dict[current_id]

        # Filtramos lenguas artificiales o "bookkeeping"
        if nodo.get("bookkeeping") or nodo.get("name") == "Unclassifiable":
            return "Unclassifiable"

        # Insertamos el nombre al inicio de la lista para mantener el orden (Raíz -> Lengua)
        linaje.insert(0, str(nodo["name"]))

        # Subimos al nodo padre
        current_id = nodo["parent_id"]

    return " > ".join(linaje)


# %%
mexico_languages = mexico_languages.copy()

mexico_languages["tree"] = mexico_languages["glottocode"].apply(reconstruir_linaje)

# %%
df_mexico = mexico_languages[
    ~mexico_languages["tree"].isin(["", "Unclassifiable"])
].copy()


# %%
df_mexico["Family"] = df_mexico["tree"].str.split().str[0]

# %%
import plotly.express as px

fig = px.scatter_geo(
    df_mexico,
    lat="latitude",
    lon="longitude",
    color="Family",
    hover_name="name",
    title="Diversidad lingüística en México"
)

fig.show()

# %% [markdown]
# ## Hacemos exactamente lo mismo para otro país, elegiremos Perú.
#

# %%
min_lat = -18
max_lat = 0
min_long = -82
max_long = -68

peru_languages = languages[
    (languages["latitude"] >= min_lat)
    & (languages["latitude"] <= max_lat)
    & (languages["longitude"] >= min_long)
    & (languages["longitude"] <= max_long)
]

# %%
peru_languages = peru_languages.copy()
peru_languages["tree"] = peru_languages["glottocode"].apply(reconstruir_linaje)
df_peru = peru_languages[
    ~peru_languages["tree"].isin(["", "Unclassifiable"])
].copy()
df_peru["Family"] = df_peru["tree"].str.split().str[0]
# %%
# Visualización de la diversidad lingüística en Perú
fig = px.scatter_geo(
    df_peru,
    lat="latitude",
    lon="longitude",
    color="Family",
    hover_name="name",
    title="Diversidad lingüística en Perú"
)
fig.show()

# %% [markdown]
# ### Diversidad lingüistica de México en comparacion a Perú
# Notemos la longitud de lenguajes en nuestras regiones

# %%
len(df_mexico)

# %%
len(df_peru)

# %%
df_mexico["Family"].nunique()

# %%
df_peru["Family"].nunique()

# %% [markdown]
# Diversidad lingüística en México con respecto a Perú
#
# México presenta una mayor cantidad de lenguas en comparación con Perú (404 frente a 177), lo que indica una alta diversidad en términos absolutos. Sin embargo, el número de familias lingüísticas es muy similar entre ambos países (36 en México y 37 en Perú), lo que sugiere que, aunque México tiene más lenguas, estas están distribuidas en un número comparable de familias.

# %%
df_mexico["Family"].value_counts()

# %% [markdown]
# ### ¿Dónde se encuentra la mayor diversidad lingüistica en méxico? 
# Zonas de mayor diversidad en México
#
# La mayor diversidad lingüística en México se concentra en el sur y centro del país, particularmente en estados como Oaxaca, Chiapas, Guerrero y Veracruz. Esto se refleja en la alta presencia de lenguas pertenecientes a familias como la otomangue, la uto-azteca y la maya, siendo la familia otomangue la más dominante con una diferencia considerable respecto a las demás.
