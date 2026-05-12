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

# %% [markdown] id="743ff36c"
# # 7. Modelos:  BPE, Embeddings, Neural LM

# %% [markdown] id="4cad4b62"
# <a target="_blank" href="https://colab.research.google.com/github/umoqnier/cl-2026-2-lab/blob/main/notebooks/7_neural_lm.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# %% [markdown] id="245b8e67"
# ## Objetivos
#
# - Entrenar modelos para sub-word tokenization
#   - Aplicar BPE a corpus
# - Entrenar modelos para *embeddings*
#   - Word2Vec
#     - Skip gram
#     - CBow
# - Implementación de modelo del lenguaje Neuronal de Bengio
#   - Generación de lenguaje

# %% id="842d9260-ec3b-4178-9b55-e0355e55dd5c"
import os
import re
from rich import print as rprint
from nltk import word_tokenize
from collections import Counter
from nltk.stem.snowball import SnowballStemmer

# %% [markdown] id="9d94c922-7b60-41db-ae50-a8c09612e476"
# ## Funciones de preprocesamiento

# %% id="de301338-c7aa-4ab6-9f73-7195781175d0"
import unicodedata


def strip_accents(s: str) -> str:
    """Remove diacritical marks from characters in a Unicode string.

    Uses Unicode NFD (Normalization Form Decomposition) normalization to decompose accented characters into their
    base character + combining mark, then filters out combining marks (Mark, Nonspacing, Mn category).
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def preprocess_text(text: str, to_lower: bool = True) -> str:
    """Preprocess text by normalizing, lowercasing and removing extra spaces.
    """
    # 1. Unicode Normalization (NFC)
    text = unicodedata.normalize("NFC", text)

    if to_lower:
        text = text.lower()

    # 3. Collapse all whitespace/newlines into a single space
    text = re.sub(r"\s+", " ", text)

    # 4. Clean up leading/trailing whitespace
    text = text.strip()

    return text


# %% [markdown] id="b606d7e6"
# ## Byte-Pair Encoding

# %% [markdown] id="39558226"
# ### Vamos a tokenizar 🌈
# ![](https://i.pinimg.com/736x/58/6b/88/586b8825f010ce0e3f9c831f568aafa8.jpg)

# %% id="ffa681bb"
BASE_PATH = "."
CORPORA_PATH = f"{BASE_PATH}/data/"
MODELS_PATH = f"{BASE_PATH}/models/"

os.makedirs(CORPORA_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)


# %% id="4dced547"
TOKENIZERS_DATA_PATH = f"{CORPORA_PATH}/tokenization"
TOKENIZERS_MODEL_PATH = f"{MODELS_PATH}/sub-word"

os.makedirs(TOKENIZERS_DATA_PATH, exist_ok=True)
os.makedirs(TOKENIZERS_MODEL_PATH, exist_ok=True)

# %% [markdown] id="f7755ef4"
# ### Corpus en español: Wikipedia

# %% id="21eb4634"
from datasets import load_dataset, load_dataset_builder

# %% colab={"base_uri": "https://localhost:8080/", "height": 208, "referenced_widgets": ["09e41e11ab9b4eba9c84b197378c5c74", "1b6a4200562c41619162f591f9fa6b3e", "a842cdd0b3a2461eb56f5877a62ac427", "a745060bcce94c5298728c46ef9c5349", "80032cfed6c744ca93f34830a9df7cb9", "af3a0e53a9f14b838813584398160ac9", "b77be1926c1244e6a459c8804bdf9db6", "64df7e129b76461c8a593ceadd2a881b", "9885c30d71554bd7a8f1100795deabb5", "1eaf33abc73649d194f8a739ab5a5d4f", "8269ddb14b0746a6bad251505537db98"]} id="637bbf74" outputId="97aa43c1-9c6b-4bca-8863-47d30cc22993"
data_builder = load_dataset_builder("wikimedia/wikipedia", "20231101.es")

# %% colab={"base_uri": "https://localhost:8080/", "height": 454} id="276090d0" outputId="8f15fc7d-a820-44a0-85d7-97bda73df58f"
rprint(data_builder.info)

# %% id="64000f67"
dataset = load_dataset(
    "wikimedia/wikipedia", "20231101.es", split="train", streaming=True
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 161} id="88f4976f" outputId="58bca1af-239e-4de8-bcff-dcace31b5901"
wiki_words = []
for article in dataset.take(1):
    rprint(preprocess_text(article["text"][:1000], to_lower=False))

# %% colab={"base_uri": "https://localhost:8080/"} id="99c2bb84" outputId="a8ec2348-3094-474f-9f1a-4fd52215e616"
# %%time

wiki_file_path = f"{TOKENIZERS_DATA_PATH}/wikipedia_es_plain.txt"
with open(wiki_file_path, "w", encoding="utf-8") as f:
    for article in dataset.take(1000):
        f.write(preprocess_text(article["text"]))
        f.write("\n")

# %% colab={"base_uri": "https://localhost:8080/"} id="0656860d" outputId="5e17410f-6293-4863-a6ce-b018425c4636"
# !head -n 10 {TOKENIZERS_DATA_PATH}/wikipedia_es_plain.txt

# %% [markdown] id="5d6e1120"
# ### Entrenando nuestro modelo con BPE
#
# ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia1.tenor.com%2Fimages%2Fd565618bb1217a7c435579d9172270d0%2Ftenor.gif%3Fitemid%3D3379322&f=1&nofb=1&ipt=9719714edb643995ce9d978c8bab77f5310204960093070e37e183d5372096d9&ipo=images)

# %% colab={"base_uri": "https://localhost:8080/"} id="f78ff367" outputId="10ccaad3-0059-4810-db24-98b43c69ef53"
# !pip install subword-nmt

# %% colab={"base_uri": "https://localhost:8080/"} id="1c42565f" outputId="524c4c94-debe-43df-fe75-fad73c05362a"
# !ls {TOKENIZERS_DATA_PATH}

# %% colab={"base_uri": "https://localhost:8080/"} id="74019265" outputId="627e8a86-6553-418d-f8c6-a073322e4bf9"
# !subword-nmt --help

# %% colab={"base_uri": "https://localhost:8080/"} id="a9aa267f" outputId="fc2f6f71-168a-468c-9bd8-83f5ff795e2b"
# !subword-nmt learn-bpe --help

# %% colab={"base_uri": "https://localhost:8080/"} id="a2695e9d" outputId="1b613548-e1cf-43ca-cdcd-5eaeb05514e4"
# %%time

# !subword-nmt learn-bpe --num-workers -1 -s 300 < \
#  {TOKENIZERS_DATA_PATH}/wikipedia_es_plain.txt > \
#   {TOKENIZERS_MODEL_PATH}/wiki_es_300.model

# %% colab={"base_uri": "https://localhost:8080/"} id="bff73459" outputId="8c683c6a-8388-4c06-be2b-60f24729994a"
# !echo "ando haciendo un análisis para claramente ver si puedes procesar esta oración mano" \
# | subword-nmt apply-bpe -c {TOKENIZERS_MODEL_PATH}/wiki_es_300.model

# %% colab={"base_uri": "https://localhost:8080/"} id="a88e2cf2" outputId="2412ddd8-d82a-4f05-93d2-637d67a0534f"
# %%time

# !subword-nmt learn-bpe --num-workers -1 -s 10000 < \
# {TOKENIZERS_DATA_PATH}/wikipedia_es_plain.txt > \
#  {TOKENIZERS_MODEL_PATH}/wiki_es_10k.model

# %% colab={"base_uri": "https://localhost:8080/"} id="b7d258bf" outputId="13957c57-2cf9-43b2-e924-646def678b6b"
# !echo "ando haciendo un análisis para claramente ver si puedes procesar esta oración mano" \
# | subword-nmt apply-bpe -c {TOKENIZERS_MODEL_PATH}/wiki_es_10k.model

# %% colab={"base_uri": "https://localhost:8080/", "height": 194, "referenced_widgets": ["14fa699db673467888354a5fb263a45a", "628f2e93d1c64d6588a3a2aa57229316", "5865b2b0f2344a2b94e478e3865d182d", "8fc64c1088c2425ebe234cdacf41be05", "cc100cc4ff98457799937f017156fd9e", "77d139fc45bc418885f9e1665e5a5394", "58553cbcf6334f0eaa1a10af3ce42b04", "9dcff1d53d18491683b729df505cbfeb", "2f127d66821e4873837a58a9512bddf1", "c4c289a4e1654f7fa00c74f6f5dcedfb", "43e852d5a3344624aff1ee99e5b55c46", "15db3a36f2914d5697aec20dd36e87d8", "811368cd27394960a0a87feb62ec7d82", "38572da85e9f4b5aa90aadec9cc1277d", "0e5ab187015e4f3191192ff2a8322f33", "e25b851bfb5f4e608756bf52713e8bb9", "a431031f165e4261b3122a18fe7f9c06", "6a0fe63f4e5f4122973c7cb7723471a4", "aa42108e77a44081a66239c8e6d03978", "e6dabe200f7f4d86997d4561a7a374f4", "7e94816c14bb45bfb729a88896e33a1a", "f03cd2728339458d8c21a51b007deef6", "20e8b6bc9d4d4c619c460f67c58394be", "24174244bab441f7b6ba9807ddc2017d", "433665c926ec472292fb924c5151b253", "56e2dd6e3f9049a6a3bcf3b0825124a1", "721fe616467941cd88862b49f283b877", "21c490cdbb5f423f8ccce7b2d9842fca", "5392b0ffb3a8407a86cde6348c5940e5", "d3304f38fc5d44fe82107b4fdf9bf4be", "5f20818877fc408e93ff154315076526", "a86b65ee743840edae735e75cc4d149f", "c722c8d883d24a8586a27e379a567195", "d0e7006ebbcc4a2da5b442d17846b168", "aaaa655f8447446fac3131ae3c37499d", "e37165e319a540b4b94b97da8b30e923", "6795f1ec456f4ed6aeadba922f53de49", "83f3e8bf36c94776862b430484d64fac", "b48de7876bc445ab8935948b2d29da1b", "af452755ea04406eaf4484d5585e7322", "360e1e3344d24a49884aa1850f54ce6b", "09fbb964930140629b6e720a10c5372d", "52c9146ba6544013b9c33407e87176a5", "075844136c204ea98c495ab5642e95ca", "cd9d579293894b849b3f58b95b86a4bf", "5e6efb3f48f74af49075f201613cc637", "9573c515499f45169e3ed037697c0199", "96fc8eae21b74e129622529d5c8c7c83", "a19b03b3f27d4ec6a44e40b20f4c14ea", "ea567f08ff2046b186f8d90c121a5f56", "90652c090024435a8f6021a6ca1ec8bc", "47adc80a53694aca9e931308bc772e81", "607730ce037648f7be7cfeac427c6c3e", "3f58fc8ed9c941ba8cc80e55e13b1902", "7f2cfd6e5d64474493eb01f8a11ed492"]} id="70e33ba2" outputId="d155465c-f13d-41f9-8977-a176870390e8"
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(
    " ".join(
        tokenizer.tokenize(
            "ando haciendo un análisis para claramente ver si puedes procesar esta oración mano"
        )
    ).replace("Ġ", "@@")
)

# %% [markdown] id="0ce450b7"
# ### Aplicandolo a otros corpus: La biblia 📖🇻🇦

# %% id="9719861b"
BIBLE_FILE_NAMES = {
    "spa": "spa-x-bible-reinavaleracontemporanea",
    "eng": "eng-x-bible-kingjames",
}

# %% id="fc67eb26"
import requests


def get_bible_corpus(lang: str) -> str:
    """Download bible file corpus from GitHub repo"""
    file_name = BIBLE_FILE_NAMES[lang]
    r = requests.get(
        f"https://raw.githubusercontent.com/ximenina/theturningpoint/main/Detailed/corpora/corpusPBC/{file_name}.txt.clean.txt"
    )
    return r.text


def write_plain_text_corpus(raw_text: str, file_name: str) -> None:
    """Write file text on disk"""
    with open(f"{file_name}.txt", "w") as f:
        f.write(raw_text)


# %% [markdown] id="3bbb273f"
# #### Tokenizando biblia en español

# %% id="fc7a4a7b"
spa_bible_raw = get_bible_corpus("spa")
spa_bible_plain_text = preprocess_text(spa_bible_raw)

# %% id="50f63dd1"
write_plain_text_corpus(spa_bible_plain_text, f"{CORPORA_PATH}/bible-spa")

# %% id="0d8dbacd"
# !subword-nmt apply-bpe -c {TOKENIZERS_MODEL_PATH}/wiki_es_10k.model < \
#  {TOKENIZERS_DATA_PATH}/bible-spa.txt > \
#  {TOKENIZERS_DATA_PATH}/bible-spa-tokenized.txt

# %% [markdown] id="5e59e5f1"
# #### Comparando resultados

# %% colab={"base_uri": "https://localhost:8080/"} id="LcKwpaRpxKzw" outputId="c1ddf4dc-0a75-4642-f283-d29d1387dce2"
import nltk
nltk.download("punkt_tab")

# %% id="c78bd45b"
spa_bible_words = word_tokenize(spa_bible_plain_text)

# %% colab={"base_uri": "https://localhost:8080/"} id="3ab3a2b4" outputId="4f8ec449-0e85-4970-abc2-b7c00d42afdd"
spa_bible_words[:10]

# %% colab={"base_uri": "https://localhost:8080/"} id="688c154f" outputId="f31d632b-6360-42d1-d2d0-728472c9e172"
len(spa_bible_words)

# %% colab={"base_uri": "https://localhost:8080/"} id="cd05ae0d" outputId="cc232c3c-61f5-4ea1-dab1-3561de80448b"
spa_bible_types = Counter(spa_bible_words)
len(spa_bible_types)

# %% colab={"base_uri": "https://localhost:8080/"} id="7a64bcbe" outputId="7b590134-606a-4e5f-a591-9a0581124ed1"
spa_bible_types.most_common(30)

# %% id="95918c90"
with open(f"{TOKENIZERS_DATA_PATH}/bible-spa-tokenized.txt", "r") as f:
    tokenized_text = f.read()
spa_bible_tokenized = tokenized_text.split()

# %% colab={"base_uri": "https://localhost:8080/"} id="335c7189" outputId="e040545e-9761-4302-98b2-37174531bc2b"
spa_bible_tokenized[:10]

# %% colab={"base_uri": "https://localhost:8080/"} id="d553ed0f" outputId="a5aa77e3-0389-48eb-f6f3-97a4ad66aa43"
len(spa_bible_tokenized)

# %% colab={"base_uri": "https://localhost:8080/"} id="b886d6bc" outputId="d4820e5b-6da1-4734-e46e-6344128112dc"
spa_bible_tokenized_types = Counter(spa_bible_tokenized)
len(spa_bible_tokenized_types)

# %% colab={"base_uri": "https://localhost:8080/"} id="e4149c6f" outputId="b0633d08-125c-4939-aa33-16c5fea100bf"
spa_bible_tokenized_types.most_common(40)

# %% colab={"base_uri": "https://localhost:8080/", "height": 65} id="3700cc19" outputId="64bda4c9-fb29-4ee3-b2fb-72f786777874"
rprint("Biblia español")
rprint(f"Tipos ([bright_magenta]word-base[/]): {len(spa_bible_types)}")
rprint(f"Tipos ([bright_green]sub-word[/]): {len(spa_bible_tokenized_types)}")

# %% [markdown] id="b1c6054a"
# #### OOV: out of vocabulary

# %% [markdown] id="54f77eaf"
# Palabras que se vieron en el entrenamiento pero no estan en el test

# %% colab={"base_uri": "https://localhost:8080/", "height": 33} id="ae88bab6" outputId="08f9decc-62fe-4070-c0c4-21955d9e68a5"
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(
    spa_bible_words, test_size=0.3, random_state=42
)
rprint(len(train_data), len(test_data))

# %% colab={"base_uri": "https://localhost:8080/", "height": 49} id="12fb0588" outputId="30e25321-7e5c-4bcf-dffa-c4df9a55b8b3"
s_1 = {"a", "b", "c", "d", "e"}
s_2 = {"a", "x", "y", "d"}
rprint(s_1 - s_2)
rprint(s_2 - s_1)

# %% id="b0a7b823"
oov_test = set(test_data) - set(train_data)

# %% colab={"base_uri": "https://localhost:8080/", "height": 65} id="4ff597ea" outputId="805d34f9-93d2-4b20-c124-5a109c3504e5"
for word in list(oov_test)[:3]:
    rprint(f"{word} in train: {word in set(train_data)}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 33} id="e5aeeb6f" outputId="f2c2d99e-7230-4f40-a88c-06ede8c73f96"
train_tokenized, test_tokenized = train_test_split(
    spa_bible_tokenized, test_size=0.3, random_state=42
)
rprint(len(train_tokenized), len(test_tokenized))

# %% id="b5bd597d"
oov_tokenized_test = set(test_tokenized) - set(train_tokenized)

# %% colab={"base_uri": "https://localhost:8080/", "height": 49} id="85387798" outputId="85521264-c75a-48e7-c1ba-9a0b234f7beb"
rprint("OOV ([bright_magenta]word-base[/]):", len(oov_test))
rprint("OOV ([bright_green]sub-word[/]):", len(oov_tokenized_test))

# %% [markdown] id="718cd1a3"
# ### Type-token Ratio (TTR)
#
# - Una forma de medir la variación del vocabulario en un corpus
# - Este se calcula como $TTR = \frac{len(types)}{len(tokens)}$
# - Puede ser útil para monitorear la variación lexica de un texto

# %% id="3ed01f83"
stemmer = SnowballStemmer("spanish")
spa_bible_stemmed = [stemmer.stem(word) for word in spa_bible_words]
spa_bible_stemmed_types = set(spa_bible_stemmed)

# %% colab={"base_uri": "https://localhost:8080/", "height": 147} id="a3860221" outputId="3df352a1-4309-4f80-df7f-9cafc401c8f6"
rprint("Bible Spanish Information")
rprint("Tokens:", len(spa_bible_words))
rprint("Types ([bright_magenta]word-base[/]):", len(spa_bible_types))
rprint("Types ([bright_yellow]stemmed[/])", len(spa_bible_stemmed_types))
rprint("Types ([bright_green]BPE[/]):", len(spa_bible_tokenized_types))
rprint(
    "TTR ([bright_magenta]word-base[/]):", len(spa_bible_types) / len(spa_bible_words)
)
rprint(
    "TTR ([bright_yellow]stemmed[/]):",
    len(spa_bible_stemmed_types) / len(spa_bible_stemmed),
)
rprint(
    "TTR ([bright_green]BPE[/]):",
    len(spa_bible_tokenized_types) / len(spa_bible_tokenized),
)

# %% [markdown] id="ec517317"
# ## Word Embeddings (W2V)

# %% [markdown] id="653ea5ef"
# Vamos a entrenar nuestras propias representaciones vectoriales utilizando la biblioteca [Gensim](https://radimrehurek.com/gensim/).

# %% colab={"base_uri": "https://localhost:8080/"} id="Jss_Di-50ASq" outputId="efe6ef23-a1aa-406d-b893-3216aeb72c36"
# !pip install gensim

# %% [markdown] id="813108d7"
# ![we](https://miro.medium.com/v2/resize:fit:2000/1*SYiW1MUZul1NvL1kc1RxwQ.png)

# %% [markdown] id="ebce77ea"
# ### Datos: Noticias en Español

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="eBsQYaUCzOJS" outputId="a2f4c306-051e-4eff-ef5d-0c1b24d7b078"
# !pip install datasets==3.6.0

# %% colab={"base_uri": "https://localhost:8080/"} id="eb2abf47" outputId="6bc24ee3-843a-433f-9b76-228a640333e3"
news_databuilder = load_dataset_builder("LeoCordoba/CC-NEWS-ES", "mx")

# %% colab={"base_uri": "https://localhost:8080/", "height": 593} id="22c64af9" outputId="3fa2f1ae-91bc-4db8-b0e5-e7ed24962a1f"
rprint(news_databuilder.info)

# %% id="8f704d2e"
news_dataset = load_dataset(
    "LeoCordoba/CC-NEWS-ES", "mx", split="train", streaming=True
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 129} id="964a5e62" outputId="35710edf-c549-40b5-d4eb-30207010fc8b"
for post in news_dataset.take(1):
    rprint(post["text"])

# %% colab={"base_uri": "https://localhost:8080/"} id="85d350c4" outputId="18ef233e-985a-45ef-fd6f-c7527111288e"
from gensim.utils import simple_preprocess

print(simple_preprocess(post["text"], deacc=True)[:10])

# %% id="a6e0fb01"
from tqdm.notebook import tqdm
from datasets import load_dataset
from gensim.utils import simple_preprocess


class CCNewsExtractor:
    """
    Iterador optimizado para CC-NEWS-ES + Word2Vec.
    Diseñado para alta velocidad y compatibilidad con los epochs de Gensim.
    """

    def __init__(self, lang: str = "mx", max_posts: int = -1):
        self.dataset = load_dataset(
            "LeoCordoba/CC-NEWS-ES", name=lang, split="train", streaming=True
        )
        self.max_posts = max_posts

    def __iter__(self):
        for item in tqdm(self.dataset.take(self.max_posts)):
            text = item.get("text", "")
            if not text:
                continue

            words = simple_preprocess(text, deacc=False, min_len=6)

            if not words:
                continue
            yield words


# %% id="084af415"
# Uso con tu función train_model
iterator = CCNewsExtractor(lang="mx", max_posts=3)

# %% colab={"base_uri": "https://localhost:8080/", "height": 449, "referenced_widgets": ["32d1947b384d49399004e303c7166018", "c15cc45accc54b28b7e546e4dfc9793c", "418257a18b6d4a369d7b2d92acac8603", "f6d0747dd56c4ab4ab88344e59e0350a", "0b4b0f903eff42978f70849fb3d247b0", "c953faa4f2e3485792bb764db42fad1d", "3b67e3f692f8458995ea52a223ff55ca", "445d5c37c5004a6e836298bae6358c7b", "8ceaaa62f9a44b92b30aaa8435bfc9b7", "bcc0689b9d6040cf90eabb41cdee38d7", "c352c19ca4f147ab8aa0f757ae0ea4e8"]} id="c7cf5886" outputId="55cc9e7d-9cfc-483e-a619-d594dc8373b8"
for i in iterator:
    rprint(i[:10])

# %% colab={"base_uri": "https://localhost:8080/"} id="6e6c27a1" outputId="72d01fd2-65da-4944-bc91-42a185bb01d0"
# %%time
sentences = CCNewsExtractor(lang="mx", max_posts=10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 243, "referenced_widgets": ["79209b7451d944f9b2fa53c422cfbb64", "ebf25cf2ec5241a58211da254c1cf52d", "45255de27d454b15b8023a407fc8fff6", "88d08afbd00742cd8647eaab80afe822", "f20213c240fd49568cd42e4b6fa76f38", "7411a6fb3fd84f4c9272a31d43019e2f", "41dec06d1fd24eec9bd9a16a590725d8", "1a4067e32d1d4320812ab8abd88ae0f3", "687ca4751f48456f9a97cd32cb6f3769", "fd9f4e8f6a894271bb91709e7ff1a8f5", "ee0b10a296f446f88751f4bf75d9c334"]} id="dc7dc74b" outputId="6c32ce58-c411-49ea-fa43-f82c23faf690"
for sentence in sentences:
    print(sentence)

# %% id="b90df0d4"
from gensim.models import word2vec

# %% id="734abc97"
EMB_MODELS_DIR = os.path.join(MODELS_PATH, "embeddings")

os.makedirs(EMB_MODELS_DIR, exist_ok=True)

# %% id="fe1aa799"
from enum import Enum


class Algorithms(Enum):
    CBOW = 0
    SKIP_GRAM = 1


# %% id="41e35849"
def load_model(model_path: str):
    """Load a word2vec model from a given path.
    """
    try:
        print(model_path)
        return word2vec.Word2Vec.load(model_path)
    except FileNotFoundError:
        print(f"[WARN] Model not found in path {model_path}")
        return None


# %% id="8880024f"
def train_model(
    sentences: list,
    model_name: str,
    vector_size: int,
    window=5,
    workers=2,
    algorithm=Algorithms.CBOW,
):
    model_name_params = f"{model_name}-vs{vector_size}-w{window}-{algorithm.name}.model"
    model_path = os.path.join(EMB_MODELS_DIR, model_name_params)
    if load_model(model_path) is not None:
        print(f"Already exists the model {model_path}")
        return load_model(model_path)
    print(f"TRAINING: {model_path}")
    if algorithm in [Algorithms.CBOW, Algorithms.SKIP_GRAM]:
        model = word2vec.Word2Vec(
            sentences,
            vector_size=vector_size,
            window=window,
            workers=workers,
            sg=algorithm.value,
            seed=42,
        )
    else:
        print("[ERROR] algorithm not implemented yet :p")
        return
    try:
        model.save(model_path)
    except:
        print(f"[ERROR] Saving model at {model_path}")
    return model


# %% [markdown] id="3cc1d7a0"
# ### CBOW

# %% colab={"base_uri": "https://localhost:8080/"} id="082df081" outputId="4d281d41-aba9-4df9-a60b-b19f0fd095dd"
skipm_gram_model = load_model(
    os.path.join(EMB_MODELS_DIR, "eswiki-xl-300-SKIP_GRAM.model")
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 281, "referenced_widgets": ["6e69c38e898749f083cc0395299aa2c9", "e2d72455a052446da97ac2d81ec8451b", "fc11044b94744ec7bc9dba8f195e8b55", "4a14c93e875b4b2c9d682df5baa535f2", "21ee267bc971419388b2fe8d7b4cbada", "1d164a7dbc9e49f79c2fa7350e40e316", "8a11d8f1895047f38d337dc88ba27751", "77f73b78659e47e2a70142e9b0187894", "382d1017547041028e73e26ed3b50ff6", "1d7639e8eeac4042b7b893f914468782", "2841aa7636a942c68c1e0250c3fd1446", "3ff52b8b1cae4b86a6324386ec91e6ed", "ea8d35bdda5c49c9857c58893a905fef", "f6793063e9f24ef1be8c7e5fd5a41bea", "8d746c80729a42679aef4050c2521069", "d526117e19c445d381df9a45bb256c6c", "3b8452caf8894132a891e9b897b32ac3", "71ac7bacda2a4cb68b5152266ad332b2", "a919339e462f4405bc31f67b7016d77f", "1ebb2f9965d84b06a7edd9410c1a50e3", "02d135788bef468c914831f23fb3df99", "9d129d3bf9644e4d81d488a2ae2c7bfe", "836583e22bbd42bea9b2ddb698e524d4", "39095308aeea4d9a84cad28d76c5effa", "a7e10aa683094643a089247b0feb35a7", "bb628c45fffa4a89b40239ccd57b22f6", "c8385ecb73e542e1897bf4b251a68352", "2c86f6505b9b4d5e9af4680cee6aae8e", "146106405e2444188c56e2e3ffdc120a", "54ee97ec58db4eec90a6221cdc7155ed", "8efef09186564bb4bde1928cb310202a", "e336e3b7b4a14e28a25716a535376f62", "bc61eb3adcdd4c85a2c3a79ab202ae4f", "6f0be09116e24da8b021ac2863b4916d", "c5c1ae9788c64da195419e591c8e59bc", "97d67a06e99f4e91921c82b8778715b1", "ffd1d45a7f234f93aff554c473e2ce8f", "49359b121ad44f3d95b1a8a1fe889c91", "19965c4560984389be2545f5d97c34cc", "de6e0fb362dd4ce4903bf02689d7be6a", "2159b2d0507c477eb63daf021a43027f", "98c9e6991bfd42ec9eec2f4a011f0019", "0335b9ebdb5544fbaa80a346f2188ba1", "c2f49049169944e2ba6cf0e97d54c7fa", "94fb7a2f497441f8a0419c73f1b9240d", "21ee6950d94e4493a5220c10a0858a64", "6a1ad197b1f34517b70af22ba8e5ca43", "c5a599de55a741cc96bed72702464877", "9034e506d13a4d98933f8cd89c86397b", "a42521f7e72b4e9fb2e34626714333b9", "7c81796fa375430e9d20d0e29106821b", "e58928276ae64947852150865a43bf46", "a425759e46c74a898bfe1773b3086d2b", "257f28259d87446193b9b6d7b94ed817", "c4753d9c71984c9e873f321476ec0663", "005f0a106f964615b72b4fb5256ff276"]} id="bf4aefb5" outputId="9666c331-eb4e-4418-83e5-66cfc6da4861"
# %%time
cbow_model = train_model(
    CCNewsExtractor(lang="mx", max_posts=100_000),
    "es_news_hf",
    vector_size=100,
    window=3,
    workers=2,
    algorithm=Algorithms.CBOW,
)

# %% [markdown] id="86141418"
# ### Skip gram

# %% id="85fbbf76"
# %%time
skip_gram_model = train_model(
    CCNewsExtractor(lang="mx", max_posts=100_000),
    "es_news_hf",
    300,
    5,
    workers=12,
    algorithm=Algorithms.SKIP_GRAM,
)


# %% id="b86ebcc3"
def report_stats(model) -> None:
    """Print report of a model"""
    print(
        "Number of words in the corpus used for training the model: ",
        model.corpus_count,
    )
    print("Number of words in the model: ", len(model.wv.index_to_key))
    print("Time [s], required for training the model: ", model.total_train_time)
    print("Count of trainings performed to generate this model: ", model.train_count)
    print("Length of the word2vec vectors: ", model.vector_size)
    print("Applied context length for generating the model: ", model.window)


# %% colab={"base_uri": "https://localhost:8080/"} id="69cc65ab" outputId="2f973567-bab2-4252-91c1-9350b3438fa2"
report_stats(cbow_model)

# %% id="4c6a76cc"
report_stats(skip_gram_model)

# %% [markdown] id="970d7489"
# ### Operaciones con los vectores entrenados
#
# Veremos operaciones comunes sobre vectores. Estos resultados dependeran del modelo que hayamos cargado en memoria

# %% id="130c02f7"
models = {
    Algorithms.CBOW: cbow_model,
    #Algorithms.SKIP_GRAM: skip_gram_model,
}

# %% id="340bdf69"
model = models[Algorithms.CBOW]

# %% colab={"base_uri": "https://localhost:8080/"} id="28469109" outputId="73029eb5-9228-4783-f910-b73d4ffc41d3"
for index, word in enumerate(model.wv.index_to_key):
    if index == 100:
        break
    print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")

# %% colab={"base_uri": "https://localhost:8080/"} id="c26b5c14" outputId="6156f60a-a065-438a-8689-a7cd6272ccd5"
gato_vec = model.wv["méxico"]
print(gato_vec[:10])
print(len(gato_vec))

# %% colab={"base_uri": "https://localhost:8080/"} id="f25250e6" outputId="37b7a140-e4d3-4493-b1b5-aeaa96362422"
try:
    agustisidad_vec = model.wv["agusticidad"]
except KeyError:
    print("OOV founded!")


# %% colab={"base_uri": "https://localhost:8080/", "height": 159} id="baf16593" outputId="dac46411-eb90-439f-a703-fc904d0c983e"
agustisidad_vec[:10]
len(agustisidad_vec)

# %% colab={"base_uri": "https://localhost:8080/"} id="1e83f265" outputId="5167f1ac-afbd-4c72-aff3-4292b17e645f"
model.wv.most_similar("méxico", topn=5)

# %% [markdown] id="9d5a6911"
# Podemos ver como la similitud entre palabras decrece

# %% colab={"base_uri": "https://localhost:8080/"} id="5fcce6eb" outputId="bd77b364-ddb5-4b5b-f57f-27579e42bb18"
word_pairs = [
    ("automóvil", "camión"),
    ("automóvil", "bicicleta"),
    ("automóvil", "cereal"),
    ("automóvil", "distrito"),
]

for w1, w2 in word_pairs:
    print(f"{w1} - {w2} {model.wv.similarity(w1, w2)}")

# %% colab={"base_uri": "https://localhost:8080/"} id="efc96d15" outputId="c5feb29c-22ce-499c-f876-be1d9121441b"
# rey es a hombre como ___ a mujer
# londres es a inglaterra como ____ a vino
model.wv.most_similar(positive=["saltillo", "morelos"], negative=["cuernavaca"])

# %% colab={"base_uri": "https://localhost:8080/", "height": 72} id="bf872fe7" outputId="6d5ea428-4875-4a0c-f462-6943a4e803de"
model.wv.doesnt_match(["disco", "música", "mantequilla", "cantante"])

# %% colab={"base_uri": "https://localhost:8080/", "height": 304} id="24f0fcbc" outputId="c023195d-95a9-4bec-eac8-beb7407015e5"
model.wv.similarity("noche", "noches")

# %% [markdown] id="dd59a663"
# #### Evaluación

# %% [markdown] id="00852465"
# `Word2Vec` es una tarea de entrenamiento semi-supervisada, por lo tanto, es difícil evaluar el desempeño de un modelo. La evaluación dependerá de la tarea.
#
# Sin embargo, Google liberó un conjunto de evaluación con ejemplos semánticos/sintácticos. Se sigue la forma "A es a B como C es a D". Por ejemplo, "tokio es a japon como berlin es a alemania".
#
# Se tienen varias categorias como comparaciones sintácticas, capitales, miembros de una familia, etc.

# %% id="64c3d6d1"
from gensim.test.utils import datapath

model.wv.evaluate_word_analogies(datapath("questions-words.txt"))


# %% [markdown] id="756707a8"
# ## Modelos del Lenguaje Neuronales (Bengio)

# %% [markdown] id="f8105f47"
# - [(Bengio et al 2003)](https://dl.acm.org/doi/10.5555/944919.944966) proponen una arquitecura neuronal como alternativa a los modelos del lenguaje estadísticos
# - Esta arquitectura lidia mejor con los casos donde las probabilidades se hacen cero, sin necesidad de aplicar una técnica de smoothing.

# %% [markdown] id="093ca909"
# <p float="left">
#   <img src="https://toppng.com/public/uploads/preview/at-the-movies-will-smith-meme-tada-11562851401lnexjqtwf9.png" width="100" />
#   <img src="https://abhinavcreed13.github.io/assets/images/bengio-model.png" width="600"/>
# </p>

# %% id="7d2a95ca"
def lm_preprocess_corpus(corpus: list[str]) -> list[str]:
    """Función de preprocesamiento para LM

    Esta función está diseñada para preprocesar
    corpus para modelos del lenguaje neuronales.
    Agrega tokens de inicio y fin, normaliza
    palabras a minusculas
    """
    preprocessed_corpus = []
    for sent in corpus:
        result = [word.lower() for word in sent]
        # Al final de la oración
        result.append("<EOS>")
        result.insert(0, "<BOS>")
        preprocessed_corpus.append(result)
    return preprocessed_corpus


# %% id="a2622ce8"
def get_words_freqs(corpus: list[list[str]]):
    """Calcula la frecuencia de las palabras en un corpus"""
    words_freqs = {}
    for sentence in corpus:
        for word in sentence:
            words_freqs[word] = words_freqs.get(word, 0) + 1
    return words_freqs


# %% id="ce93db1e"
UNK_LABEL = "<UNK>"


def get_words_indexes(words_freqs: dict) -> dict:
    """Calcula los indices de las palabras dadas sus frecuencias"""
    result = {}
    for idx, word in enumerate(words_freqs.keys()):
        # Happax legomena happends
        if words_freqs[word] == 1:
            # Temp index for unknowns
            result[UNK_LABEL] = len(words_freqs)
        else:
            result[word] = idx

    return {word: idx for idx, word in enumerate(result.keys())}, {
        idx: word for idx, word in enumerate(result.keys())
    }


# %% colab={"base_uri": "https://localhost:8080/"} id="8467bd75" outputId="05d7b225-c5e1-496b-a961-1c0cc8364b14"
import nltk

nltk.download("gutenberg")
nltk.download("abc")
nltk.download("genesis")
nltk.download("inaugural")
nltk.download("state_union")
nltk.download("webtext")
nltk.download("punkt_tab")

# %% colab={"base_uri": "https://localhost:8080/"} id="dffa0b36" outputId="5b3cfcb8-908a-4205-fdbe-2db9c9927721"
from nltk.corpus import abc, genesis, gutenberg, inaugural, state_union, webtext

# Exploración del corpus
total_sents = 0
corpora = []

plaintext_corpora = {
    "abc": abc,
    "Gutenberg": gutenberg,
    #"Genesis": genesis, Este no lo usamos por una buena razón
    "Inaugural": inaugural,
    "State Union": state_union,
    "Web": webtext,
}

for title, corpus in plaintext_corpora.items():
    corpus_sents = lm_preprocess_corpus(corpus.sents())
    corpus_len = len(corpus_sents)
    total_sents += corpus_len
    print(f"{title} sents={corpus_len}")
    corpora.extend(corpus_sents)
print(f"Total={total_sents}")

# %% colab={"base_uri": "https://localhost:8080/"} id="261420b8" outputId="b014e654-1409-4888-8552-2dc1ac2b0dbb"
len(corpora)

# %% colab={"base_uri": "https://localhost:8080/"} id="0fdd04cc" outputId="336ad93a-11a9-445f-b9ff-cab6f79911da"
corpora[42]

# %% id="96f216c7"
words_freqs = get_words_freqs(corpora)

# %% colab={"base_uri": "https://localhost:8080/"} id="630467d8" outputId="cc674b55-4c0c-466a-f33e-c831912add27"
words_freqs["the"]

# %% colab={"base_uri": "https://localhost:8080/"} id="25f3e872" outputId="bddaf33c-7d24-4e12-cd32-b48ffc73d4bc"
len(words_freqs)

# %% colab={"base_uri": "https://localhost:8080/"} id="f145eb99" outputId="41411d70-c398-4703-f489-5b2f7ae8e8b9"
count = 0
for word, freq in words_freqs.items():
    if freq == 1 and count <= 10:
        print(word, freq)
        count += 1

# %% id="63a16e13"
words_indexes, index_to_word = get_words_indexes(words_freqs)

# %% colab={"base_uri": "https://localhost:8080/"} id="ed0c79f7" outputId="864f1f45-519c-4f5c-daf4-d5ce8965a94f"
words_indexes["god"]

# %% colab={"base_uri": "https://localhost:8080/", "height": 34} id="e06d7a70" outputId="a962409b-7308-4fef-c74a-91400b2ae627"
index_to_word[9562]

# %% colab={"base_uri": "https://localhost:8080/"} id="df7181ea" outputId="6ee994b6-1fd1-417b-96f3-02396a3f23ea"
len(words_indexes)

# %% colab={"base_uri": "https://localhost:8080/"} id="82cb219f" outputId="77aa7b61-2634-4aa3-b810-f62f26c2b649"
len(index_to_word)


# %% id="96d717ea"
def get_word_id(words_indexes: dict, word: str) -> int:
    """Obtiene el id de una palabra dada

    Si no se encuentra la palabra se regresa el id
    del token UNK
    """
    unk_word_id = words_indexes[UNK_LABEL]
    return words_indexes.get(word, unk_word_id)


# %% [markdown] id="15273362"
# ### Obtenemos trigramas

# %% [markdown] id="7ec26cbd"
# Convertiremos los trigramas obtenidos a secuencias de idx, y preparamos el conjunto de entrenamiento $x$ y $y$
#
# - x: Contexto
# - y: Predicción de la siguiente palabra

# %% id="7072ef69"
from nltk import ngrams


def get_train_test_data(
    corpus: list[list[str]], words_indexes: dict, n: int
) -> tuple[list, list]:
    """Obtiene el conjunto de train y test

    Requerido en el step de entrenamiento del modelo neuronal
    """
    x_train = []
    y_train = []
    for sent in corpus:
        n_grams = ngrams(sent, n)
        for w1, w2, w3 in n_grams:
            x_train.append(
                [get_word_id(words_indexes, w1), get_word_id(words_indexes, w2)]
            )
            y_train.append([get_word_id(words_indexes, w3)])
    return x_train, y_train


# %% [markdown] id="5557091b"
# ### Preparando Pytorch
#
# $x' = e(x_1) \oplus e(x_2)$
#
# $h = \tanh(W_1 x' + b)$
#
# $y = softmax(W_2 h)$

# %% id="a7a48db5"
# cargamos bibliotecas
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# %% id="7192261e"
# Setup de parametros
EMBEDDING_DIM = 200
CONTEXT_SIZE = 2
BATCH_SIZE = 256

H = 100
torch.manual_seed(42)
# Tamaño del Vocabulario
V = len(words_indexes)

# %% id="9ce4924f"
x_train, y_train = get_train_test_data(corpora, words_indexes, n=3)

# %% id="2cdd6dc5"
import numpy as np

train_set = np.concatenate((x_train, y_train), axis=1)
# partimos los datos de entrada en batches
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)


# %% [markdown] id="7514f9fb"
# ### Creamos la arquitectura del modelo

# %% id="bcee7728"
# Trigram Neural Network Model
class TrigramModel(nn.Module):
    """Clase padre: https://pytorch.org/docs/stable/generated/torch.nn.Module.html"""

    def __init__(self, vocab_size, embedding_dim, context_size, h):
        super(TrigramModel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        # TODO: Se aprenden los embeddings de aca?
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, h)
        self.linear2 = nn.Linear(h, vocab_size)

    def forward(self, inputs):
        # x': concatenation of x1 and x2 embeddings   -->
        # self.embeddings regresa un vector por cada uno de los índices que se les pase como entrada.
        # view() les cambia el tamaño para concatenarlos
        embeds = self.embeddings(inputs).view(
            (-1, self.context_size * self.embedding_dim)
        )
        # h: tanh(W_1.x' + b)  -->
        out = torch.tanh(self.linear1(embeds))
        # W_2.h                 -->
        out = self.linear2(out)
        # log_softmax(W_2.h)      -->
        # dim=1 para que opere sobre renglones, pues al usar batchs tenemos varios vectores de salida
        log_probs = F.log_softmax(out, dim=1)

        return log_probs


# %% id="11887c70"
# Seleccionar la GPU si está disponible
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 34} id="qS00COpN_Zqf" outputId="5428f777-1eaf-41fc-e979-19cb3865193b"
device

# %% id="4911347a"
import os
NN_MODELS_PATH = os.path.join("models", "nn")

os.makedirs(NN_MODELS_PATH, exist_ok=True)

LM_PATH = os.path.join(NN_MODELS_PATH, "trigrams_nlm_cpu_epoch3.pt")

# %% colab={"base_uri": "https://localhost:8080/", "height": 439} id="1226c934" outputId="40f6cc1e-9a83-4749-a89a-b283a889f276"
print(f"Training on device {device}")

# 1. Pérdida. Negative log-likelihood loss
loss_function = nn.NLLLoss()

# 2. Instanciar el modelo y enviarlo a device
model = TrigramModel(V, EMBEDDING_DIM, CONTEXT_SIZE, H).to(device)

# 3. Optimización. ADAM optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-3)

# ------------------------- TRAIN & SAVE MODEL ------------------------
EPOCHS = 5
for epoch in range(EPOCHS):
    st = time.time()
    print("\n--- Training model Epoch: {} ---".format(epoch))
    for it, data_tensor in enumerate(train_loader):
        # Mover los datos al dispositivo
        context_tensor = data_tensor[:, 0:2].to(device)
        target_tensor = data_tensor[:, 2].to(device)

        # Resetamos los gradientes de la iteración anterior
        model.zero_grad()

        # FORWARD:
        log_probs = model(context_tensor)

        # compute loss function
        loss = loss_function(log_probs, target_tensor)

        # BACKWARD:
        loss.backward()
        optimizer.step()

        if it % 500 == 0:
            print(
                "Training Iteration {} of epoch {} complete. Loss: {}; Time taken (s): {}".format(
                    it, epoch, loss.item(), (time.time() - st)
                )
            )
            st = time.time()

    # saving model
    model_path = os.path.join(
        NN_MODELS_PATH, f"lm_large_{device}_context_{CONTEXT_SIZE}_epoch_{epoch}.dat"
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved for epoch={epoch} at {model_path}")


# %% colab={"base_uri": "https://localhost:8080/"} id="b09b6123" outputId="e609aaed-463e-4a23-96bd-9ca9abb4f93a"
model


# %% id="fd942e1b"
def get_torch_model(path: str) -> TrigramModel:
    """Obtiene modelo de pytorch desde disco"""
    model_loaded = TrigramModel(V, EMBEDDING_DIM, CONTEXT_SIZE, H)
    model_loaded.load_state_dict(torch.load(path))
    model_loaded.eval()
    return model_loaded


# %% id="45fb6565"
model = get_torch_model(os.path.join("models/nn/", "lm_large_cuda_context_2_epoch_0.dat"))

# %% colab={"base_uri": "https://localhost:8080/", "height": 34} id="xsgSDqEzB0xV" outputId="86560e54-1590-46fa-dbd4-2fdcf3886a64"
device

# %% id="53d25930"
W1 = "<BOS>"
W2 = "my"

IDX1 = get_word_id(words_indexes, W1)
IDX2 = get_word_id(words_indexes, W2)

# Obtenemos Log probabidades p(W3|W2,W1)
probs = model(torch.tensor([[IDX1, IDX2]]).to("cpu")).detach().tolist()

# %% colab={"base_uri": "https://localhost:8080/"} id="77b2fec8" outputId="4d5630b0-f246-4102-81b8-3bb99eaf4484"
len(probs[0])

# %% colab={"base_uri": "https://localhost:8080/"} id="8456319b" outputId="75836b35-ea8d-4911-c38c-b7dedd9d6d3f"
# Creamos diccionario con {idx: logprob}
model_probs = {}
for idx, p in enumerate(probs[0]):
    model_probs[idx] = p

# Sort:
model_probs_sorted = sorted(
    ((prob, idx) for idx, prob in model_probs.items()), reverse=True
)

# Printing word  and prob (retrieving the idx):
topcandidates = 0
for prob, idx in model_probs_sorted:
    # Retrieve the word associated with that idx
    word = index_to_word[idx]
    print(idx, word, prob, np.exp(prob))

    topcandidates += 1

    if topcandidates > 10:
        break

# %% colab={"base_uri": "https://localhost:8080/"} id="baa2c6f8" outputId="e18e8918-a137-4173-a75f-7cb0b1957747"
print(index_to_word.get(model_probs_sorted[0][1]))


# %% [markdown] id="7c38cad1"
# ### Generacion de lenguaje

# %% id="b3482b65"
def get_likely_words(
    model: TrigramModel,
    context: str,
    words_indexes: dict,
    index_to_word: dict,
) -> list[tuple]:
    """Dado un contexto obtiene las palabras más probables
    """
    model_probs = {}
    words = context.split()
    idx_word_1 = get_word_id(words_indexes, words[0])
    idx_word_2 = get_word_id(words_indexes, words[1])
    probs = model(torch.tensor([[idx_word_1, idx_word_2]]).to("cpu")).detach().tolist()

    for idx, p in enumerate(probs[0]):
        model_probs[idx] = p

    # Strategy: Sort and get top-K words to generate text
    return sorted(
        ((prob, index_to_word[idx]) for idx, prob in model_probs.items()), reverse=True
    )


# %% colab={"base_uri": "https://localhost:8080/"} id="9dec1f7a" outputId="0501ad34-8a77-4376-b613-05bcd3c1706f"
sentence = "this is"
get_likely_words(model, sentence, words_indexes, index_to_word)[:3]

# %% id="49ab4b69"
import random
from random import randint

def get_next_top_p_word(words: list[tuple[float, str]], p: float = 0.8) -> str:
    """
    Selecciona la siguiente palabra utilizando Nucleus Sampling (Top-p).

    Params:
    - words: Lista de tuplas (palabra, probabilidad).
    - p: Umbral de masa de probabilidad acumulada (típicamente entre 0.8 y 0.95).
    """
    if not words:
        return "<EOS>"

    # Aseguramos que la lista esté ordenada de mayor a menor probabilidad
    # sorted_words = sorted(words, key=lambda x: x[1], reverse=True)

    valid_words = []
    valid_probs = []
    cumulative_prob = 0.0

    # Recolectamos palabras hasta que la suma de probabilidades alcance el umbral 'p'
    for log_prob, word in words:
        # Convertimos log_prob a probabilidad normal
        prob = np.exp(log_prob)
        valid_words.append(word)
        valid_probs.append(prob)
        cumulative_prob += prob

        if cumulative_prob >= p:
            break

    # Muestreamos una palabra del subconjunto válido (núcleo) usando sus probabilidades como pesos.
    # random.choices devuelve una lista, por lo que extraemos el elemento [0]
    return random.choices(valid_words, weights=valid_probs, k=1)[0]


def get_next_word(words: list[tuple[float, str]]) -> str:
    # From a top-K list of words get a random word
    return words[randint(0, len(words) - 1)][1]


# %% colab={"base_uri": "https://localhost:8080/", "height": 34} id="89312b33" outputId="09048708-b121-453c-b82e-c68f47ec37ad"
get_next_top_p_word(get_likely_words(model, sentence, words_indexes, index_to_word))

# %% id="pYUZOx8zC0Sk"

# %% id="08771b8c"
MAX_TOKENS = 50
TOP_P = 0.7


def generate_text(
    model: TrigramModel,
    history: str,
    words_indexes: dict,
    index_to_word: dict,
    tokens_count: int = 0,
) -> None:
    next_word = get_next_top_p_word(
        get_likely_words(
            model, history, words_indexes, index_to_word
        ), p=TOP_P
    )
    print(next_word, end=" ")
    tokens_count += 1
    if tokens_count == MAX_TOKENS or next_word == "<EOS>":
        return
    generate_text(
        model,
        history.split()[1] + " " + next_word,
        words_indexes,
        index_to_word,
        tokens_count,
    )


# %% colab={"base_uri": "https://localhost:8080/"} id="a6023c72" outputId="c6e52291-8758-47a9-ad70-b038a07cd306"
sentence = "god tells"
print(sentence, end=" ")
generate_text(model, sentence, words_indexes, index_to_word)

# %% [markdown] id="910edc51"
# # Práctica 4: Evaluación de modelos del lenguaje neuronales
#
# **Fecha: 5 de Mayo 2026 11:59pm**
#
# ### Formáto de entrega
# - Crear una carpeta con el nombre de su equipo dentro de `practicas/`
# - Incluir los archivos requeridos (notebook, script Python, README)
# - Ejemplo de estructura:
#
# ```
# practicas/
# ├── krustaceo/
# │   └── P4
# │       ├── mi_practica4.ipynb
# │       ├── mi_practica4.py
# │       └── README.md  # <-- Incluir los nombres de los integrantes
# ```
#
# #### Investigación
#
# La calidad de un modelo del lenguaje puede ser evaluado por medio de su perplejidad (perplexity)
#
# - Investigar como calcular la perplejidad de un modelo del lenguaje y como evaluarlo con esa medida
#     - Incluir en el `README.md` de su entrega una síntesis de esta investigación. Sean breves
#         - Explicación clara de qué es la **perplejidad** (perplexity) y cómo se calcula
#         - Fórmula matemática con explicación de cada componente
#         - Relación entre perplejidad y calidad del modelo
#         - Ventajas y limitaciones de esta métrica
# - Evalua el modelo entrenado en clase con los corpus de `nltk`
#     - Descarga el modelo acá
#
# #### Creación de modelos del lenguaje
#
# - Entrena un nuevo modelo del lenguaje neuronal con los corpus de `nltk` aplicando previamente sub-word tokenization a los corpus
#     - Puedes utilizar un modelo de tokenizacion pre-entrenado o entrenar uno desde cero
#     - Utiliza el corpus `genesis` de `nltk` como test de evaluación.
# - Evalua tu modelo calculando su perplejidad.
#
#
# #### Análisis comparativo
#
# - Realizar un análisis comparativo entre ambos modelos.
#
# | Métrica               | Modelo Base | Modelo Subword |
# |-----------------------|-------------|----------------|
# | Perplejidad (genesis) |             |                |
# | Tamaño vocabulario    |             |                |
# | OOV Rate              |             |                |
#
# - Incluir en el `README.md`:
#     - Discusión sobre qué modelo tuvo mejor desempeño y por qué
#     - Ventajas y desventajas de cada enfoque
#     - Recomendaciones para mejorar ambos modelos
#
#
# **NOTA:** Sube tu modelo a alguna plataforma de almacenamiento (Google Drive, Nextcloud, Hugging Face, etc), proporciona el link de descarga y el código para cargar el modelo en memoria. **No subas tu modelo al repositorio de GitHub**.
#
# ## EXTRA
#
# - Diseña una estrategia de generación de usando el modelo del lenguaje entrenado con sub-word tokenization
# - Se deben generar secuencias de palabras (no subwords)
# - Muestra tres ejemplos de generación
