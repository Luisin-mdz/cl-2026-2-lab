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
# # 1.1 Niveles Lingüísticos I

# + [markdown] editable=true slideshow={"slide_type": ""}
# <a target="_blank" href="https://colab.research.google.com/github/umoqnier/cl-2026-2-lab/blob/main/notebooks/1_niveles_linguisticos_I.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# + [markdown] editable=true id="615a09ab-2b52-440a-a4dc-fd8982c3c0e7" slideshow={"slide_type": "subslide"}
# ## Objetivos

# + [markdown] editable=true id="f034458c-9cb0-4966-a203-3145074c3fca" slideshow={"slide_type": ""}
# - Trabajar tareas a diferentes niveles lingüísticos (Fonético, Morfólogico, Sintáctico)
# - Manipular y recuper información de datasets disponibles en Github para resolver tareas de NLP
# - Comparar enfoques basados en reglas y estadísticos para el análisis morfológico

# + [markdown] editable=true id="3c169487-91d2-4afb-a12a-849c26a5be86" slideshow={"slide_type": "subslide"}
# ## Fonética y Fonología

# + [markdown] editable=true id="d0647e1e-a8c5-418f-81c7-31d2e86c88a4" slideshow={"slide_type": ""}
# <center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/IPA_chart_2020.svg/660px-IPA_chart_2020.svg.png" width=500></center

# + editable=true slideshow={"slide_type": "subslide"}
from IPython.display import YouTubeVideo

# + editable=true slideshow={"slide_type": ""}
YouTubeVideo("DcNMCB-Gsn8", width=960, height=615)

# + editable=true slideshow={"slide_type": "subslide"}
YouTubeVideo("74nnLh0Vdcc", width=960, height=615)

# + editable=true slideshow={"slide_type": "subslide"}
import http
from collections import defaultdict

import pandas as pd
import requests as r

from rich import print as rprint
from rich.columns import Columns
from rich.panel import Panel
from rich.text import Text

# + [markdown] editable=true id="aa915e8e-038e-4de6-8956-f6221b1d8490" slideshow={"slide_type": "subslide"}
# ### International Phonetic Alphabet (IPA)

# + [markdown] editable=true id="09b4f076-b23b-46a8-9101-e37d79d374c8" slideshow={"slide_type": ""}
# - Las lenguas naturales tienen muchos sonidos diferentes por lo que necesitamos una forma de describirlos independientemente de las lenguas
# - IPA es una representación escrita de los [sonidos](https://www.ipachart.com/) del [habla](http://ipa-reader.xyz/)

# + [markdown] editable=true id="19eee353-6fd4-474a-86ca-8382ad51bf0f" slideshow={"slide_type": "subslide"}
# ### Dataset: [IPA-dict](https://github.com/open-dict-data/ipa-dict) de open-dict

# + [markdown] editable=true id="18f45a54-5f64-408e-98f3-fc31114dc84a" slideshow={"slide_type": "fragment"}
# - Diccionario de palabras para varios idiomas con su representación fonética
# - Representación simple, una palabra por renglon con el formato:
#
# ```
# [PALABRA][TAB][IPA]
#
# Ejemplos
# mariguana	/maɾiɣwana/
# zyuganov's   /ˈzjuɡɑnɑvz/, /ˈzuɡɑnɑvz/
# ```

# + [markdown] editable=true id="7cb52e47-d493-4b30-a991-ba5c4458d047" slideshow={"slide_type": "fragment"}
# - [ISO language codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
# - URL: `https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/<iso-lang>`

# + [markdown] editable=true id="4152e020-fbc0-4ec5-8d51-ccd2e8a089fc" slideshow={"slide_type": "subslide"}
# #### Explorando el corpus 🗺️

# + editable=true id="25b595d7-7201-42bd-abb3-3acf9731d219" slideshow={"slide_type": "fragment"}
IPA_URL = "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/{lang}.txt"

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="3f45ba75-bbd3-4f13-8abf-b822fbf90dda" outputId="fa1dcad8-0851-4dad-eedd-49f1d91db7cb" slideshow={"slide_type": "fragment"}
# ¿Como empezamos?
response = r.get(IPA_URL.format(lang="es_MX"))
# -

ipa_list = response.text[:1000].split("\n")

ipa_list[-1].split("\t")


# + [markdown] editable=true id="c671dbe4-1f99-443a-afb9-3f92951bef35" slideshow={"slide_type": "subslide"}
# #### Obtención y manipulación

# + editable=true id="1fdc23af-9a0b-470d-a5f1-e5bddfa0b53e" slideshow={"slide_type": "fragment"}
def download_ipa_corpus(iso_lang: str) -> str:
    """Get ipa-dict file from Github

    Parameters:
    -----------
    iso_lang:
        Language as iso code

    Results:
    --------
    dict:
        Dictionary with words as keys and phonetic representation
        as values for a given lang code
    """
    print(f"Downloading {iso_lang}", end="::")
    response = r.get(IPA_URL.format(lang=iso_lang))
    status_code = response.status_code
    print(f"status={status_code}")
    if status_code != http.HTTPStatus.OK:
        print(f"ERROR on {iso_lang} :(")
        return ""
    return response.text
# + editable=true slideshow={"slide_type": "subslide"}
download_ipa_corpus("ar")[:100]


# + editable=true id="0a83a2a2-8e0e-4881-98f3-b9251a6be778" slideshow={"slide_type": "subslide"}
def parse_response(response: str) -> dict:
    """Parse text response from ipa-dict to python dict

    Each row have the format:
    [WORD][TAB]/[IPA]/(, /[IPA]/)?

    Parameters
    ----------
    response: str
        ipa-dict raw text

    Returns
    -------
    dict:
        A dictionary with the word as key and the phonetic
        representations as value
    """
    ipa_list = response.rstrip().split("\n")
    result = {}
    for item in ipa_list:
        if item == '':
            continue
        item_list = item.split("\t")
        result[item_list[0]] = item_list[1]
    return result
# + editable=true slideshow={"slide_type": "subslide"}
es_data = parse_response(download_ipa_corpus("es_MX"))


# + editable=true id="b834aaba-0716-41b5-935b-7f4a61e9da03" slideshow={"slide_type": "subslide"}
def get_ipa_transcriptions(word: str, dataset: dict) -> list[str]:
    """Search for a word in an IPA phonetics dict

    Given a word this function return the IPA transcriptions

    Parameters:
    -----------
    word: str
        A word to search in the dataset
    dataset: dict
        A dataset for a given language code

    Returns
    -------
    list[str]:
        List with posible transcriptions if any,
        else an empty list
    """
    return dataset.get(word.lower(), "").split(", ")


# + colab={"base_uri": "https://localhost:8080/"} editable=true id="d89e45e2-5010-4701-84fd-e62b910233e7" outputId="0839146b-ed76-4c0a-c7cf-078ac7278791" slideshow={"slide_type": "subslide"}
get_ipa_transcriptions("mayonesa", es_data)


# + [markdown] editable=true id="37f69f04-ad55-4ca2-8bb1-d52fa58c4051" slideshow={"slide_type": "subslide"}
# #### Obtengamos datasets

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="0828cb63-19b9-4cac-8df2-7dc8282fc4c3" outputId="763fe008-d810-4822-b1fe-826b784b988a" slideshow={"slide_type": "fragment"}
# Get datasets
dataset_es_mx = parse_response(download_ipa_corpus("es_MX"))
dataset_ja = parse_response(download_ipa_corpus("ja"))
dataset_en_us = parse_response(download_ipa_corpus("en_US"))
dataset_fr = parse_response(download_ipa_corpus("fr_FR"))

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="694ef1c1-871e-407e-9ce5-5e177804f72f" outputId="f221b632-e77c-4649-e194-dbb49e2a644b" slideshow={"slide_type": "subslide"}
# Simple query
get_ipa_transcriptions("beautiful", dataset_en_us)

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="04a5fdbb-3acc-4e34-9685-61a591f2b598" outputId="13068823-f467-4526-d9d9-32b3f5bcf340" slideshow={"slide_type": "fragment"}
# Examples
print(f"dog -> {get_ipa_transcriptions('dog', dataset_en_us)} 🐶")
print(f"mariguana -> {get_ipa_transcriptions('mariguana', dataset_es_mx)} 🪴")
print(f"猫 - > {get_ipa_transcriptions('猫', dataset_ja)} 🐈")
print(f"croissant -> {get_ipa_transcriptions('croissant', dataset_fr)} 🥐")

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="9001ea35-855f-499b-a5b9-3c70a0ba7397" outputId="91960028-7806-4544-99ba-af5ea517c68e" slideshow={"slide_type": "fragment"}
# Diferentes formas de pronunciar
print(f"[es_MX] hotel | {dataset_es_mx['hotel']}")
print(f"[en_US] hotel | {dataset_en_us['hotel']}")

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="9fc9c19c-f6f4-4e35-9155-71aacaef1a05" outputId="16852ef9-832b-42ca-9390-717c42104153" slideshow={"slide_type": "fragment"}
print(f"[ja] ホテル | {dataset_ja['ホテル']}")
print(f"[fr] hôtel | {dataset_fr['hôtel']}")


# + [markdown] editable=true id="41ca5bf8-93b1-4b10-9596-02bce9caccb8" slideshow={"slide_type": "subslide"}
# #### Obtener la distribución de frecuencias de los símbolos fonológicos para español

# + id="d3868dcd-eb1b-4e4a-9f52-ad7e2be72971"
def get_phone_symbols_freq(dataset: dict) -> defaultdict[str, int]:
    freqs = defaultdict(int)
    ipas = [_.strip("/") for _ in dataset.values()]
    unique_ipas = set(ipas)
    for ipa in unique_ipas:
        for char in ipa:
            freqs[char] += 1
    return freqs


# + id="ijDBSOM5UB5i"
freqs_es = get_phone_symbols_freq(dataset_ja)
# Sorted by freq number (d[1]) descendent (reverse=True)
distribution_es = dict(sorted(freqs_es.items(), key=lambda d: d[1], reverse=True))
df_es = pd.DataFrame.from_dict(distribution_es, orient='index')
# -

df_es.head()

# + [markdown] editable=true id="aeb6c269-ad75-4e49-9090-247dc9a60231" slideshow={"slide_type": "slide"}
# #### 🧙🏼‍♂️ Ejercicio: Encontrar homófonos (palabras con el mismo sonido pero distinta ortografía) para el español
#
# - Ejemplos: Casa-Caza, Vaya-Valla

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="-UXEnSv6700t" outputId="0e02caa7-93da-4e37-c304-f1b96073f44d" slideshow={"slide_type": "fragment"}
from collections import Counter

transcription_counts = Counter(dataset_es_mx.values())
duplicated_transcriptions = [
    transcription for transcription, freq in transcription_counts.items() if freq > 1
]

for ipa in duplicated_transcriptions[-10:]:
    words = [
        word for word, transcription in dataset_es_mx.items() if transcription == ipa
    ]
    rprint(f"{ipa} => {words}")

# +
# Counter?

# + [markdown] editable=true id="a6e06a95-ceb6-49c0-bcbb-ff456976e510" slideshow={"slide_type": "subslide"}
# #### Obteniendo todos los datos

# + editable=true id="9d92e8bd-53c9-4f2a-b926-b2de8ac19357" slideshow={"slide_type": "subslide"}
lang_codes = {
    "ar": "Arabic (Modern Standard)",
    "de": "German",
    "en_UK": "English (Received Pronunciation)",
    "en_US": "English (General American)",
    "eo": "Esperanto",
    "es_ES": "Spanish (Spain)",
    "es_MX": "Spanish (Mexico)",
    "fa": "Persian",
    "fi": "Finnish",
    "fr_FR": "French (France)",
    "fr_QC": "French (Québec)",
    "is": "Icelandic",
    "ja": "Japanese",
    "jam": "Jamaican Creole",
    "km": "Khmer",
    "ko": "Korean",
    "ma": "Malay (Malaysian and Indonesian)",
    "nb": "Norwegian Bokmål",
    "nl": "Dutch",
    "or": "Odia",
    "ro": "Romanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tts": "Isan",
    "vi_C": "Vietnamese (Central)",
    "vi_N": "Vietnamese (Northern)",
    "vi_S": "Vietnamese (Southern)",
    "yue": "Cantonese",
    "zh_hans": "Mandarin (Simplified)",
    "zh_hant": "Mandarin (Traditional)"
}
iso_lang_codes = list(lang_codes.keys())


# + editable=true id="aaf29cd0-be3c-4821-a608-71275da4852e" slideshow={"slide_type": "subslide"}
def get_corpora() -> dict:
    """Download corpora from ipa-dict github

    Given a list of iso lang codes download available datasets.

    Returns
    -------
    dict
        Lang codes as keys and dictionary with words-transcriptions
        as values
    """
    return {
        code: parse_response(download_ipa_corpus(code))
         for code in iso_lang_codes
        }


# + colab={"base_uri": "https://localhost:8080/"} editable=true id="f04bc065-584b-4373-bc7f-9e328793bbe4" outputId="9195f012-6ba8-4c12-b0ed-1ed2f419a266" slideshow={"slide_type": "fragment"}
corpora = get_corpora()


# + [markdown] editable=true id="41cc36cd-7919-4372-ada2-d9f52432b56c" slideshow={"slide_type": "subslide"}
# #### Sistema de búsqueda (naïve)

# + editable=true id="24424a3c-70df-416f-8923-6bcd2a435007" slideshow={"slide_type": "subslide"}
def get_formated_string(code: str, name: str):
    return f"[b]{name}[/b]\n[yellow]{code}"


# + colab={"base_uri": "https://localhost:8080/", "height": 1000} editable=true id="bd09c65f-c559-4571-98a8-20c0d2c0308e" outputId="14a4e15e-43b2-4f83-8081-f5562967c8d1" slideshow={"slide_type": "fragment"}
rprint(
    Panel(Text("Representación fonética de palabras", style="bold", justify="center"))
)
rendable_langs = [
    Panel(get_formated_string(code, lang), expand=True)
    for code, lang in lang_codes.items()
]
rprint(Columns(rendable_langs))

lang = input("lang>> ")
rprint(f"Selected language: {lang_codes[lang]}") if lang else rprint("Adios 👋🏼")
while lang:
    dataset = corpora[lang]
    query = input(f"  [{lang}]word>> ")
    results = get_ipa_transcriptions(query, dataset)
    print(query, " | ", ", ".join(results))
    while query:
        query = input(f"  [{lang}]word>> ")
        if query:
            results = get_ipa_transcriptions(query, dataset)
            rprint(query, " | ", ", ".join(results))
    lang = input("lang>> ")
    rprint(f"Selected language: [yellow]{lang_codes[lang]}[/]") if lang else rprint(
        "Adios 👋🏼"
    )

# + [markdown] editable=true id="dc8b18ff-9d70-49a6-98c1-7feb7d0c268a" slideshow={"slide_type": "slide"}
# #### 👩‍🔬 Ejercicio: *[Ortographic Depth](https://en.wikipedia.org/wiki/Orthographic_depth)*
#
# Algunas lenguas se escriben como suenan y otras no. Calcula el promedio de los *ratios* (`len(chars) / len(phones)`) para todas las lenguas en el corpus. La salida debe verse más o menos así:
#
# ```
# Φ Spanish (Spain) = ??
# Φ Spanish (Mexico) = ??
# Φ English (General American) = ??
# ```
#
# Dónde `??` deberá mostrar el ratio (`Φ`) calculado
# + editable=true slideshow={"slide_type": "fragment"}
import numpy as np


def calculate_orthographic_depth(dataset):
    ratios = []
    for word, ipa in dataset.items():
        clean_ipa = ipa.strip("/").replace("ˈ", "")
        # Ratio: Letras por Sonido
        if len(clean_ipa) > 0:
            ratios.append(len(word) / len(clean_ipa))
    return np.mean(ratios)


for iso, dataset in corpora.items():
    rprint(f"Φ {lang_codes[iso]} = {calculate_orthographic_depth(dataset):.3f}")




# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# ##### ¿Pregunta?: ¿Qué lengua usarías si quieres reducir costos en un LLM?

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### Ejemplo: Fabricando rimas

# + editable=true id="a4116005-8fb4-454d-a2ab-e2f083eda000" slideshow={"slide_type": "fragment"}
def get_rhyming_patterns(sentence: str, dataset: dict) -> dict[str, list]:
    words = sentence.split()
    word_ipa_map = {}
    for word in words:
        ipa_transcriptions = get_ipa_transcriptions(word, dataset)
        # Remove "/" char from transcriptions
        word_ipa_map.update({word: [_.strip("/") for _ in ipa_transcriptions]})

    rhyming_patterns = defaultdict(list)
    for word, ipas in word_ipa_map.items():
        for ipa in ipas:
            # Getting last 2 elements of the ipa representation
            pattern = ipa[-2:]
            rhyming_patterns[pattern].append(word)
    return rhyming_patterns


# + editable=true slideshow={"slide_type": "fragment"}
def display_rhyming_patterns(patterns: dict[str, list]) -> None:
    for pattern, words in patterns.items():
        if len(set(words)) > 1:
            print(f"{pattern}:: {', '.join(words)}")


# + [markdown] editable=true id="f4daacb3-544b-4b3a-ab77-23b2fc4dd07e" slideshow={"slide_type": "subslide"}
# #### Testing

# + [markdown] editable=true id="3HoQEf8i8qTo" slideshow={"slide_type": "fragment"}
# ```
# ɣo:: juego, fuego
# on:: con, corazón
# ʎa:: brilla, orilla
# ```

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="057bb91d-5bef-47b4-ba82-42559f457c2b" outputId="0435b8cf-4395-438b-a78f-0f872b9cb287" slideshow={"slide_type": "fragment"}
sentence = "cuando juego con fuego siento como brilla la orilla de mi corazón"

dataset = corpora.get("es_MX")
rhyming_words = get_rhyming_patterns(sentence, dataset)
display_rhyming_patterns(rhyming_words)

# + [markdown] editable=true id="86f07e0f-bfe4-4a14-be9d-a7b0b7661260" slideshow={"slide_type": "subslide"}
# #### Material extra (fonética)

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="64dc5e71-449b-4fdf-a3f4-d0f1776c5bbf" outputId="8b70a454-65b1-447d-aee3-73b1933b134b" slideshow={"slide_type": ""}
# apt-get install -y espeak
# !sudo pacman -S espeak-ng
# -

# !espeak  --voices

# + editable=true id="c87a1b9d-848c-488f-b2e4-1782f07bc557" slideshow={"slide_type": ""}
# !espeak  -v roa/es-419 "Camara banda ya se la saben celulares y carteras" --ipa
# -


