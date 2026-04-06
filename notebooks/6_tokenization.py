# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 6. Preprocesamiento y tokenización

# %% [markdown]
# <img src="https://2.bp.blogspot.com/-oDvCIkIjwXw/VdWWxfvmq3I/AAAAAAAARUE/r0MrmbNzMz8/s1600/inputoutput.jpg" width=500>

# %% [markdown]
# ## Objetivos

# %% [markdown]
# - Listar algunos pasos comúnes para el preprocesamiento de texto
#   - Aplicar preprocesamiento a corpus en español e inglés
# - Entender el funcionamiento de algoritmos de sub-word tokenization
#   - BPE
#   - Word-piece
#   - Sentecepiece
# - Entrenar modelos para sub-word tokenization
#   - Aplicar BPE a corpus

# %% [markdown]
# ## Elementos del preprocesamiento

# %% [markdown]
# - Limpieza del texto
#     - Quitar etiquetas de marcado (HTML, XML, MD), metadatos y asegurarnos que todo esta en UTF-8
#     - Eliminar header, footers o titulos que no aportan información
# - Normalización
#     - Pasar todo a minúsculas
#     - Pasar texto a cierta norma ortográfica
#     - Expansión de contracciones o abreviaciones
# - Quitar stopwords y lematización/stemming
# - Tokenización
#     - Por palabra
#     - Por letras
#     - Por sub-palabras
# - *Embeddings*
#     - Los modelos solo entienden números, por lo que hay que convertir el texto a números
#     - Cada palabra se representa como un vector de números 

# %% [markdown]
# ### Stopwords

# %%
import re
import nltk
from nltk.corpus import stopwords
from rich import print as rprint

# %%
BASE_PATH = "./"
CORPORA_PATH = f"{BASE_PATH}/corpora/tokenization"
MODELS_PATH = f"{BASE_PATH}/models/sub-word"

# %%
nltk.download('stopwords')

# %%
rprint(stopwords.words("spanish")[:15])

# %% [markdown]
# ### Normalización

# %% [markdown]
# <center><img src="https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fimg1.wikia.nocookie.net%2F__cb20140504152558%2Fspongebob%2Fimages%2Fe%2Fe3%2FThe_spongebob.jpg&f=1&nofb=1&ipt=28368023b54a7c84c9100025981b1042d0f4ca3ceaac53be42094cc1c3794348&ipo=images" height=300 width=300></center>

# %%
import unicodedata

def strip_accents(s: str) -> str:
   return ''.join(
       c for c in unicodedata.normalize('NFD', s)
       if unicodedata.category(c) != 'Mn'
   )


# %%
strip_accents("mamá hoy quería que me oigan en el olímpo")


# %% [markdown]
# - https://www.unicode.org/reports/tr44/#GC_Values_Table
#
# > And keep in mind, these manipulations may significantly alter the meaning of the text. Accents, Umlauts etc. are not "decoration".
# - [oefe](https://stackoverflow.com/users/49793/oefe) - [source](https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string)

# %%
def preprocess(words: list[str], regex: str="\w+", lang: str="en") -> list[str]:
    """Preprocess step for corpus

    Parameters
    ----------
    words: list[str]
        Words of a given corpus
    regex: str
        Optional regex to filter patterns in words. Default \w+
    lang: str
        Optional lang for choice stopwords. Default "en"

    Return
    ------
    list:
        List of words filtered and normalized

    """
    stop_lang = "english" if lang=="en" else "spanish"
    result = []
    for word in words:
        word = re.sub(f"[^\w\s]", "", word).lower()
        if word.isalpha():
            result.append(word)
    return result


# %% [markdown]
# #### ¿Para otras lenguas?

# %% [markdown]
# - No hay muchos recursos :(
# - Pero para el nahuatl esta `pyelotl` :)

# %% [markdown]
# #### Normalizando el Nahuatl

# %%
# !pip install elotl

# %%
import elotl.corpus
import elotl.nahuatl.orthography

# %%
axolotl = elotl.corpus.load("axolotl")

# %%
# Tres posibles normalizadores: sep, inali, ack
# Sauce: https://pypi.org/project/elotl/

nahuatl_normalizer = elotl.nahuatl.orthography.Normalizer("sep")

# %%
axolotl[1][1]

# %%
nahuatl_normalizer.normalize(axolotl[1][1])

# %%
nahuatl_normalizer.to_phones(axolotl[1][1])

# %% [markdown]
# ## ¿Cuántas palabras hay en las siguientes oraciones?

# %%
sentence = "Quitan el trapo y no lo ponen. ¿Por qué quitan el trapo? Si es una cosa que debe estar ahí."

# %%
sentence = "Mmmmm haz lo que quieras... pero no me digas que no te lo advertí 😓"

# %% [markdown]
# - A estas alturas tenemos cierta información acerca de las palabras:
#     - **typos:** Número de palabras únicas en un corpus. *AKA* vocabulario
#     - **tokens:** Número total de palabras. *AKA* instancias

# %% [markdown]
# ## ¿Qué es una palabra?

# %% [markdown]
# - Técnicas de procesamiento del lenguaje depende de las palabras y las oraciones.
#   - Debemos identificar estos elementos para poder procesarlos
# - Este paso de identificación de palabras y oraciones se le llama segmentación de texto o **tokenización** (*tokenization*)
# - Además de la identificación de unidades aplicaremos transformaciones al texto

# %% [markdown]
# ### Más que mil palabras

# %% [markdown]
# Aunque la definición de lo que es una palabra puede parecer obvia a la hora de diseñar sistemas de PLN puede ser tremendamente difícil.

# %% [markdown]
# - I'm
# - we'd
# - I've
# - Diego's Bicycle

# %% [markdown]
# En lenguas donde los espacios no son utilizados para marcar posibles delimitaciones entre palabras la cosa se pone más dura:
#
# - 姚明进入总决赛 - yáo míng jìn rù zong jué sài
# > Yao Ming llegó a las finales

# %% [markdown]
# Chinese Treebank:
#
# 1. 姚明 - Yao Ming
# 2. 进入 - llego a
# 3. 总决赛 - finales

# %% [markdown]
# Peking University:
#
# 1. 姚 - Yao
# 2. 明 - Ming
# 3. 进入 - llego
# 4. 总 - generales
# 5. 决赛 - finales

# %% [markdown]
# Caracteres como límites
#
# 1. 姚 - Yao
# 2. 明 - Ming
# 3. 进 - entrar
# 4. 入 - entrar
# 5. 总 - generales
# 6. 决 - decisión
# 7. 赛 - juego

# %% [markdown]
# Otro problema a considerar es la cantidad de palabras con la que tendran que lidiar los modelos que diseñemos. Por más texto que tengamos a disposición siempre habrán palabras que el modelo no habrá visto (*AKA* **Out of Vocabulary, OOV** o **\<UNK\>**)

# %% [markdown]
# ### Recordando los morfemas

# %% [markdown]
# - Con la morfología podemos identificar como se modifica el significado variando la estructura de las palabras
# - Tambien las reglas para producir:
#     - niño -> niños
#     - niño -> niña
# - Tenemos elementos mínimos, intercambiables que varian el significado de las palabras: **morfemas**
#
# > Un morfema es la unidad mínima con significado en la producción lingüística (Mijangos, 2020)

# %% [markdown]
# #### Tipos de morfemas

# %% [markdown]
# - Bases: Subcadenas que aportan información léxica de la palabra
#     - sol
#     - frasada
# - Afijos: Subcadenas que se adhieren a las bases para añadir información (flexiva, derivativa)
#     - Prefijos
#         - *in*-parable
#     - Subfijos
#         - pan-*ecitos*, come-*mos*

# %% [markdown]
# ## Tokenización

# %% [markdown]
# ### Word-base tokenization

# %%
text = """
¡¡¡Mamá prendele a la grabadora!!!, ¿llamaste a las vecinas? Corre la voz porque, efectivamente, !estoy en la tele! 📺
"""

# %%
text.split()

# %%
import re

# [a-zA-Z_]
regex = r"\w+"
re.findall(regex, text)

# %%
re.findall(regex, "El valor de PI es 3.14159")

# %% [markdown]
# <img src="http://images.wikia.com/battlebears/images/2/2c/Troll_Problem.jpg" with="250" height="250">

# %% [markdown]
# - Vocabularios gigantescos difíciles de procesar
# - Generalmente, entre más grande es el vocabulario más pesado será nuestro modelo
#
# **Ejemplo:**
# - Si queremos representaciones vectoriales de nuestras palabras obtendríamos vectores distintos para palabras similares
#     - niño = `v1(39, 34, 5,...)`
#     - niños = `v2(9, 4, 0,...)`
#     - niña = `v3(2, 1, 1,...)`
#     - ...
# - Tendríamos tokens con bajísima frecuencia
#     - merequetengue = `vn(0,0,1,...)`

# %% [markdown]
# ### Una solución: Steaming/Lematización (AKA la vieja confiable)

# %% [markdown]
# ![](https://i.pinimg.com/736x/77/df/89/77df89e6ff57d332ba4e5d7bff723133--meme.jpg)

# %%
import nltk
from nltk.corpus import brown
nltk.download('brown')

# %%
brown_corpus = preprocess(brown.words()[:100000])
rprint(brown_corpus[0])

# %%
rprint(brown_corpus[:10])

# %%
from collections import Counter

rprint(f"[yellow]Brown Vanilla")
rprint("Tokens:", len(brown.words()))
rprint("Tipos:", len(Counter(brown.words())))

rprint(f"[green]Brown Preprocess")
rprint("Tokens:", len(brown_corpus))
rprint("Tipos:", len(Counter(brown_corpus)))

# %% [markdown]
# #### Steamming

# %%
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

# %%
stemmed_brown = [stemmer.stem(word) for word in brown_corpus]

# %% [markdown]
# #### Lematización

# %%
# !python -m spacy download en_core_web_md
# !python -m spacy download es_core_news_md

# %%
import spacy

def lemmatize(words: list, lang: str="en") -> list:
    model = "en_core_web_md" if lang == "en" else "es_core_news_md"
    nlp = spacy.load(model)
    nlp.max_length = 1500000
    lemmatizer = nlp.get_pipe("lemmatizer")
    return [token.lemma_ for token in nlp(" ".join(words))]


# %%
lemmatized_brown = lemmatize(brown_corpus)

# %%
from rich.panel import Panel

rprint("Tipos ([blue]word-based):", len(Counter(brown_corpus)))
rprint("Tipos ([yellow]Steamming):", len(Counter(stemmed_brown)))
rprint("Tipos ([green]Lemmatized):", len(Counter(lemmatized_brown)))

# %% [markdown]
# #### More problems?
#
# <img src="https://uploads.dailydot.com/2019/10/Untitled_Goose_Game_Honk.jpeg?auto=compress%2Cformat&ixlib=php-3.3.0" width="250" height="250">

# %% [markdown]
# - Métodos dependientes de las lenguas
# - Se pierde información
# - Ruled-based

# %% [markdown]
# ## Subword-tokenization salva el día 🦸🏼‍♀️

# %% [markdown]
# ![](https://gifdb.com/images/high/super-cow-and-chicken-daxvak1q16quwd9p.webp)

# %% [markdown]
# - Segmentación de palabras en unidades más pequeñas (*sub-words*)
# - Obtenemos tipos menos variados y con mayores frecuencias
#     - Esto le gusta modelos basados en métodos estadísticos
# - Palabras frecuentes no deberían separarse
# - Palabras largas y raras debería descomponerse en sub-palabras significativas
# - Los métodos estadisticos que no requieren conocimiento a priori de las lenguas

# %%
text = "Let's do tokenization!"
result = ["Let's", "do", "token", "ization", "!"]
print(f"Objetivo: {text} -> {result}")

# %% [markdown]
# ### Algoritmos

# %% [markdown]
# Existen varios algoritmos para hacer *subword-tokenization* como los que se listan a continuación:
#
# - Byte-Pair Encoding (BPE)
# - WordPiece
# - Unigram

# %% [markdown]
# #### BPE

# %% [markdown]
# - Segmenmentación iterativa, comienza segmentando en secuencias de caracteres
# - Junta los pares más frecuentes (*merge operation*)
# - Termina cuando se llega al número de *merge operations* especificado o número de vocabulario deseado (*hyperparams*, depende de la implementación)
# - Introducido en el paper: [Neural Machine Translation of Rare Words with Subword Units, (Sennrich et al., 2015)](https://arxiv.org/abs/1508.07909)

# %%
# %%HTML
<iframe width="960" height="515" src="https://www.youtube.com/embed/HEikzVL-lZU"></iframe>

# %%
# !pip install transformers

# %%
SENTENCE = "Let's do this tokenization to enable hypermodernization on my tokens tokenized 👁️👁️👁️!!!"

# %%
from transformers import GPT2Tokenizer

bpe_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
rprint(bpe_tokenizer.tokenize(SENTENCE))

# %%
encoded_tokens = bpe_tokenizer(SENTENCE)
rprint(encoded_tokens["input_ids"])

# %%
rprint(bpe_tokenizer.decode(encoded_tokens["input_ids"]))

# %% [markdown]
# - En realidad GPT-2 usa *Byte-Level BPE*
#     - Evitamos vocabularios de inicio grandes (Ej: unicode)
#     - Usamos bytes como vocabulario base
#     - Evitamos *Out Of Vocabulary, OOV* (aka `[UKW]`)

# %% [markdown]
# #### WordPiece

# %% [markdown]
# - Descrito en el paper: [Japanese and Korean voice search, (Schuster et al., 2012) ](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)
# - Similar a BPE, inicia el vocabulario con todos los caracteres y aprende los merges
# - En contraste con BPE, no elige con base en los pares más frecuentes si no los pares que maximicen la probabilidad de aparecer en los datos una vez que se agregan al vocabulario
#
# $$score(a_i,b_j) = \frac{f(a_i,b_j)}{f(a_i)f(b_j)}$$
#
# - Esto quiere decir que evalua la perdida de realizar un *merge* asegurandoce que vale la pena hacerlo
#
# - Algoritmo usado en `BERT`

# %%
# %%HTML
<iframe width="960" height="500" src="https://www.youtube.com/embed/qpv6ms_t_1A"></iframe>

# %%
from transformers import BertTokenizer
SENTENCE = "🌽" + SENTENCE + "🔥"
wp_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
rprint(wp_tokenizer.tokenize(SENTENCE))

# %% [markdown]
# <center><img src="https://us-tuna-sounds-images.voicemod.net/9cf541d2-dd7f-4c1c-ae37-8bc671c855fe-1665957161744.jpg"></center>

# %%
rprint(wp_tokenizer(SENTENCE))

# %% [markdown]
# #### Unigram

# %% [markdown]
# - Algoritmo de subpword tokenization introducido en el paper: [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (Kudo, 2018)](https://arxiv.org/pdf/1804.10959.pdf)
# - En contraste con BPE o WordPiece, este algoritmo inicia con un vocabulario muy grande y va reduciendolo hasta llegar tener un vocabulario deseado
# - En cada iteración se calcula la perdida de quitar cierto elemento del vocabulario
#     - Se quitará `p%` elementos que menos aumenten la perdida en esa iteración
# - El algoritmo termina cuando se alcanza el tamaño deseado del vocabulario

# %% [markdown]
# Sin embargo, *Unigram* no se usa por si mismo en algun modelo de Hugging Face:
# > "Unigram is not used directly for any of the models in the transformers, but it’s used in conjunction with SentencePiece." - Hugging face guy

# %% [markdown]
# #### SentencePiece
#

# %% [markdown]
# - No asume que las palabras estan divididas por espacios
# - Trata la entrada de texto como un *stream* de datos crudos. Esto incluye al espacio como un caractér a usar
# - Utiliza BPE o Unigram para construir el vocabulario

# %%
# https://github.com/google/sentencepiece#installation
# !pip install sentencepiece

# %%
from transformers import XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
rprint(tokenizer.tokenize(SENTENCE))

# %% [markdown]
# #### Objetivo de los subword tokenizers
#

# %% [markdown]
# - Buscamos que modelos de redes neuronales tenga datos mas frecuentes
# - Esto ayuda a que en principio "aprendan" mejor
# - Reducir el numero de tipos
# - Reducir el numero de OOV

# %% [markdown]
# ### Vamos a tokenizar 🌈
# ![](https://i.pinimg.com/736x/58/6b/88/586b8825f010ce0e3f9c831f568aafa8.jpg)

# %% [markdown]
# #### Corpus en español: CESS

# %%
nltk.download("cess_esp")

# %%
from nltk.corpus import cess_esp

cess_words = cess_esp.words()

# %%
" ".join(cess_words[:30])

# %%
cess_plain_text = " ".join(preprocess(cess_words))

# %%
rprint(f"'{cess_plain_text[300:600]}'")

# %%
cess_preprocessed_words = cess_plain_text.split()

# %%
with open(f"{CORPORA_PATH}/cess_plain.txt", "w") as f:
    f.write(cess_plain_text)

# %% [markdown]
# #### Corpus Inglés: Gutenberg

# %%
nltk.download('gutenberg')
nltk.download("punkt_tab")

# %%
from nltk.corpus import gutenberg

gutenberg_words = gutenberg.words()[:200000]

# %%
rprint(" ".join(gutenberg_words[:30]))

# %%
gutenberg_plain_text = " ".join(preprocess(gutenberg_words))

rprint(gutenberg_plain_text[:100])

# %%
gutenberg_preprocessed_words = gutenberg_plain_text.split()

# %%
with open(f"{CORPORA_PATH}/gutenberg_plain.txt", "w") as f:
    f.write(gutenberg_plain_text)

# %% [markdown]
# #### Tokenizando el español con Hugging face

# %%
from transformers import AutoTokenizer

spanish_tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
rprint(spanish_tokenizer.tokenize(cess_plain_text[1000:1400]))

# %%
cess_types = Counter(cess_words)

# %%
rprint(cess_types.most_common(10))

# %%
cess_tokenized = spanish_tokenizer.tokenize(cess_plain_text)
rprint(cess_tokenized[:10])
cess_tokenized_types = Counter(cess_tokenized)

# %%
rprint(cess_tokenized_types.most_common(30))

# %%
cess_lemmatized_types = Counter(lemmatize(cess_words, lang="es"))

# %%
rprint(cess_lemmatized_types.most_common(30))

# %%
rprint("CESS")
rprint(f"Tipos ([blue]word-base): {len(cess_types)}")
rprint(f"Tipos ([yellow]lemmatized): {len(cess_lemmatized_types)}")
rprint(f"Tipos ([green]sub-word): {len(cess_tokenized_types)}")

# %% [markdown]
# #### Tokenizando para el inglés

# %%
gutenberg_types = Counter(gutenberg_words)

# %%
gutenberg_tokenized = wp_tokenizer.tokenize(gutenberg_plain_text)
gutenberg_tokenized_types = Counter(gutenberg_tokenized)

# %%
rprint(gutenberg_tokenized_types.most_common(100))

# %%
gutenberg_lemmatized_types = Counter(lemmatize(gutenberg_preprocessed_words))

# %%
rprint(gutenberg_lemmatized_types.most_common(20))

# %%
rprint("Gutenberg")
rprint(f"Tipos ([blue]word-base): {len(gutenberg_types)}")
rprint(f"Tipos ([yellow]lemmatized): {len(gutenberg_lemmatized_types)}")
rprint(f"Tipos ([green]sub-word): {len(gutenberg_tokenized_types)}")

# %% [markdown]
# #### OOV: out of vocabulary

# %% [markdown]
# Palabras que se vieron en el entrenamiento pero no estan en el test

# %%
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(gutenberg_words, test_size=0.3, random_state=42)
rprint(len(train_data), len(test_data))

# %%
s_1 = {"a", "b", "c", "d", "e"}
s_2 = {"a", "x", "y", "d"}
rprint(s_1 - s_2)
rprint(s_2 - s_1)

# %%
oov_test = set(test_data) - set(train_data)

# %%
for word in list(oov_test)[:3]:
    rprint(f"{word} in train: {word in set(train_data)}")

# %%
train_tokenized, test_tokenized = train_test_split(gutenberg_tokenized, test_size=0.3, random_state=42)
rprint(len(train_tokenized), len(test_tokenized))

# %%
oov_tokenized_test = set(test_tokenized) - set(train_tokenized)

# %%
rprint("OOV ([yellow]word-base):", len(oov_test))
rprint("OOV ([green]sub-word):", len(oov_tokenized_test))

# %% [markdown]
# ## Entrenando nuestro modelo con BPE
# ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia1.tenor.com%2Fimages%2Fd565618bb1217a7c435579d9172270d0%2Ftenor.gif%3Fitemid%3D3379322&f=1&nofb=1&ipt=9719714edb643995ce9d978c8bab77f5310204960093070e37e183d5372096d9&ipo=images)

# %%
# !pip install subword-nmt

# %%
# !ls {CORPORA_PATH}

# %%
# !cat {CORPORA_PATH}/gutenberg_plain.txt

# %%
# !subword-nmt learn-bpe -s 300 < \
#  {CORPORA_PATH}/gutenberg_plain.txt > \
#   {MODELS_PATH}/gutenberg.model

# %%
# !echo "I need to process this sentence because tokenization can be useful" \
# | subword-nmt apply-bpe -c {MODELS_PATH}/gutenberg.model

# %%
# !subword-nmt learn-bpe -s 1500 < \
# {CORPORA_PATH}/gutenberg_plain.txt > \
#  {MODELS_PATH}/gutenberg_high.model

# %%
# !echo "I need to process this sentence because tokenization can be useful" \
# | subword-nmt apply-bpe -c {MODELS_PATH}/gutenberg_high.model

# %% [markdown]
# ### Aplicandolo a otros corpus: La biblia 📖🇻🇦

# %%
BIBLE_FILE_NAMES = {"spa": "spa-x-bible-reinavaleracontemporanea", "eng": "eng-x-bible-kingjames"}

# %%
import requests

def get_bible_corpus(lang: str) -> str:
    """Download bible file corpus from GitHub repo"""
    file_name = BIBLE_FILE_NAMES[lang]
    r = requests.get(f"https://raw.githubusercontent.com/ximenina/theturningpoint/main/Detailed/corpora/corpusPBC/{file_name}.txt.clean.txt")
    return r.text

def write_plain_text_corpus(raw_text: str, file_name: str) -> None:
    """Write file text on disk"""
    with open(f"{file_name}.txt", "w") as f:
        f.write(raw_text)


# %% [markdown]
# #### Biblia en Inglés

# %%
eng_bible_plain_text = get_bible_corpus("eng")
eng_bible_words = eng_bible_plain_text.lower().replace("\n", " ").split()

# %%
print(eng_bible_words[:10])

# %%
len(eng_bible_words)

# %%
eng_bible_types = Counter(eng_bible_words)

# %%
rprint(eng_bible_types.most_common(30))

# %%
eng_bible_lemmas_types = Counter(lemmatize(eng_bible_words, lang="en"))

# %%
write_plain_text_corpus(eng_bible_plain_text, f"{CORPORA_PATH}/eng-bible")

# %%
# !subword-nmt apply-bpe -c {MODELS_PATH}/gutenberg_high.model < \
#  {CORPORA_PATH}/eng-bible.txt > \
#  {CORPORA_PATH}/eng-bible-tokenized.txt

# %%
with open(f"{CORPORA_PATH}/eng-bible-tokenized.txt", 'r') as f:
    tokenized_data = f.read()
eng_bible_tokenized = tokenized_data.split()

# %%
rprint(eng_bible_tokenized[:10])

# %%
len(eng_bible_tokenized)

# %%
eng_bible_tokenized_types = Counter(eng_bible_tokenized)
len(eng_bible_tokenized_types)

# %%
eng_bible_tokenized_types.most_common(30)

# %% [markdown]
# #### ¿Qué pasa si aplicamos el modelo aprendido con Gutenberg a otras lenguas?

# %%
spa_bible_plain_text = get_bible_corpus('spa')
spa_bible_words = spa_bible_plain_text.replace("\n", " ").lower().split()

# %%
spa_bible_words[:10]

# %%
len(spa_bible_words)

# %%
spa_bible_types = Counter(spa_bible_words)
len(spa_bible_types)

# %%
spa_bible_types.most_common(30)

# %%
spa_bible_lemmas_types = Counter(lemmatize(spa_bible_words, lang="es"))
len(spa_bible_lemmas_types)

# %%
write_plain_text_corpus(spa_bible_plain_text, f"{CORPORA_PATH}/spa-bible")

# %%
# !subword-nmt apply-bpe -c {MODELS_PATH}/gutenberg_high.model < \
#  {CORPORA_PATH}/spa-bible.txt > \
#  {CORPORA_PATH}/spa-bible-tokenized.txt

# %%
with open(f"{CORPORA_PATH}/spa-bible-tokenized.txt", "r") as f:
    tokenized_text = f.read()
spa_bible_tokenized = tokenized_text.split()

# %%
spa_bible_tokenized[:10]

# %%
len(spa_bible_tokenized)

# %%
spa_bible_tokenized_types = Counter(spa_bible_tokenized)
len(spa_bible_tokenized_types)

# %%
spa_bible_tokenized_types.most_common(40)

# %% [markdown]
# ### Type-token Ratio (TTR)
#
# - Una forma de medir la variazión del vocabulario en un corpus
# - Este se calcula como $TTR = \frac{len(types)}{len(tokens)}$
# - Puede ser útil para monitorear la variación lexica de un texto

# %%
rprint("Información de la biblia en Inglés")
rprint("Tokens:", len(eng_bible_words))
rprint("Types ([blue]word-base):", len(eng_bible_types))
rprint("Types ([yellow]lemmatized)", len(eng_bible_lemmas_types))
rprint("Types ([green]BPE):", len(eng_bible_tokenized_types))
rprint("TTR ([blue]word-base):", len(eng_bible_types)/len(eng_bible_words))
rprint("TTR ([green]BPE):", len(eng_bible_tokenized_types)/len(eng_bible_tokenized))

# %%
rprint("Bible Spanish Information")
rprint("Tokens:", len(spa_bible_words))
rprint("Types ([blue]word-base):", len(spa_bible_types))
rprint("Types ([yellow]lemmatized)", len(spa_bible_lemmas_types))
rprint("Types ([green]BPE):", len(spa_bible_tokenized_types))
rprint("TTR ([blue]word-base):", len(spa_bible_types)/len(spa_bible_words))
rprint("TTR ([green]BPE):", len(spa_bible_tokenized_types)/len(spa_bible_tokenized))
