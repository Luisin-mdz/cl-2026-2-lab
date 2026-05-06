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
#     display_name: cl-test
#     language: python
#     name: cl-test
# ---

# %% [markdown]
# # Práctica 4: Evaluación de modelos de lenguaje neuronales.
# Empezaremos por reutilizar todo el código necesario de la práctica 7_neural_lm.ipynb como los modelos entrenados.

# %% [markdown]
# ## 1. Código Reutilizado (Práctica Base)
# Las siguientes celdas contienen la configuración, funciones de preprocesamiento y arquitectura del modelo original desarrollado previamente.

# %%
import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from nltk import ngrams
import nltk
from nltk.corpus import abc, genesis, gutenberg, inaugural, state_union, webtext
import torch.optim as optim

# %%
nltk.download("gutenberg", quiet=True)
nltk.download("abc", quiet=True)
nltk.download("genesis", quiet=True)
nltk.download("inaugural", quiet=True)
nltk.download("state_union", quiet=True)
nltk.download("webtext", quiet=True)

# %%
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Usando dispositivo: {device}")

# %% [markdown]
# ### Cambiar esta ruta por la ruta local donde se tengan los modelos.

# %%
MODELS_PATH = "/home/luis/cl-test/notebooks/models/" 
NN_MODELS_PATH = os.path.join(MODELS_PATH, "nn")
os.makedirs(NN_MODELS_PATH, exist_ok=True)

# %%
UNK_LABEL = "<UNK>"

# %%
def lm_preprocess_corpus(corpus: list[str]) -> list[str]:
    preprocessed_corpus = []
    for sent in corpus:
        result = [word.lower() for word in sent]
        result.append("<EOS>")
        result.insert(0, "<BOS>")
        preprocessed_corpus.append(result)
    return preprocessed_corpus

# %%
def get_words_freqs(corpus: list[list[str]]):
    words_freqs = {}
    for sentence in corpus:
        for word in sentence:
            words_freqs[word] = words_freqs.get(word, 0) + 1
    return words_freqs

# %%
def get_words_indexes(words_freqs: dict) -> dict:
    result = {}
    for idx, word in enumerate(words_freqs.keys()):
        if words_freqs[word] == 1:
            result[UNK_LABEL] = len(words_freqs)
        else:
            result[word] = idx
    return {word: idx for idx, word in enumerate(result.keys())}, {idx: word for idx, word in enumerate(result.keys())}

# %%
def get_word_id(words_indexes: dict, word: str) -> int:
    """Obtiene el id de una palabra dada. Si no existe, regresa el id de <UNK>"""
    unk_word_id = words_indexes[UNK_LABEL]
    return words_indexes.get(word, unk_word_id)

def get_train_test_data(corpus: list[list[str]], words_indexes: dict, n: int) -> tuple[list, list]:
    """Obtiene el conjunto de train y test separando contexto (x) y target (y)"""
    x_train = []
    y_train = []
    for sent in corpus:
        n_grams = ngrams(sent, n)
        for w1, w2, w3 in n_grams:
            x_train.append([get_word_id(words_indexes, w1), get_word_id(words_indexes, w2)])
            y_train.append([get_word_id(words_indexes, w3)])
    return x_train, y_train

class TrigramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, h):
        super(TrigramModel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, h)
        self.linear2 = nn.Linear(h, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((-1, self.context_size * self.embedding_dim))
        out = torch.tanh(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def get_torch_model(path: str, vocab_size: int, emb_dim: int, ctx_size: int, h_size: int) -> TrigramModel:
    """Carga los pesos de un modelo guardado en disco"""
    model_loaded = TrigramModel(vocab_size, emb_dim, ctx_size, h_size)
    model_loaded.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model_loaded.eval()
    return model_loaded

# %% [markdown]
# ## 2. Reconstrucción del Vocabulario de Entrenamiento
# Recreamos el diccionario exacto utilizado durante el entrenamiento del modelo base, excluyendo explícitamente el corpus de prueba (`genesis`).

# %%
corpora = []
plaintext_corpora = {
    "abc": abc,
    "Gutenberg": gutenberg,
    "Inaugural": inaugural,
    "State Union": state_union,
    "Web": webtext,
}

print("Procesando corpus de entrenamiento.")
for title, corpus in plaintext_corpora.items():
    corpora.extend(lm_preprocess_corpus(corpus.sents()))

words_freqs = get_words_freqs(corpora)
words_indexes, index_to_word = get_words_indexes(words_freqs)

EMBEDDING_DIM = 200
CONTEXT_SIZE = 2
H = 100
V = len(words_indexes)

print(f"Tamaño total (V): {V}")

# %% [markdown]
# ## 3. Evaluación del Modelo Base (Palabras Completas)
# A continuación evaluamos el rendimiento del modelo original utilizando la tasa de OOV y la Perplejidad sobre el corpus no visto `genesis`.

# %%
test_corpus = lm_preprocess_corpus(genesis.sents())
test_words = [word for sent in test_corpus for word in sent]
total_test_tokens = len(test_words)

# %% [markdown]
# ### Cálculo del OOV rate

# %%
train_vocab = set(words_indexes.keys())
oov_tokens = sum(1 for word in test_words if word not in train_vocab)
oov_rate = oov_tokens / total_test_tokens

print("--- Evaluación de Vocabulario (Modelo Base) ---")
print(f"Tamaño del vocabulario: {len(train_vocab)}")
print(f"Tokens OOV en test: {oov_tokens}")
print(f"OOV Rate: {oov_rate:.4%} ({oov_rate:.4f})")

# %% [markdown]
# ### Calculo de la perplejidad del modelo base.

# %%
import math
from torch.utils.data import DataLoader, TensorDataset

def evaluate_perplexity(model, test_corpus, words_indexes, device, batch_size=512):
    model.eval() 
    loss_function = nn.NLLLoss(reduction='sum')
    
    x_test, y_test = get_train_test_data(test_corpus, words_indexes, n=3)
    
    x_tensor = torch.tensor(x_test)
    y_tensor = torch.tensor(y_test).view(-1)
    
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    total_loss = 0.0
    total_samples = len(x_test)
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            log_probs = model(x_batch)
            loss = loss_function(log_probs, y_batch)
            total_loss += loss.item()
            
    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss

base_model_path = os.path.join(NN_MODELS_PATH, "lm_large_cpu_context_2_epoch_0.dat")
modelo_base = get_torch_model(base_model_path, V, EMBEDDING_DIM, CONTEXT_SIZE, H)
ppl_base, avg_loss_base = evaluate_perplexity(modelo_base, test_corpus, words_indexes, device)

print("--- Evaluación de Perplejidad (Modelo Base) ---")
print(f"Pérdida (Cross-Entropy): {avg_loss_base:.4f}")
print(f"Perplejidad: {ppl_base:.4f}")

# %% [markdown]
# ## 4. Preparación de Datos con Sub-words (BPE)
# Utilizaremos el tokenizador pre-entrenado de GPT-2 basado en Byte-Pair Encoding (BPE).

# %%
from transformers import AutoTokenizer

# Cargamos el tokenizador pre-entrenado que vimos en la P7
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Agregamos nuestros tokens especiales para que no los parta a la mitad
tokenizer.add_special_tokens({'bos_token': '<BOS>', 'eos_token': '<EOS>', 'unk_token': '<UNK>'})

def subword_preprocess_corpus(corpus: list[str]) -> list[int]:
    tokenized_corpus = []
    for sent in corpus:
        text = " ".join(sent).lower()
        # Le pegamos los tokens de inicio y fin
        text = f"<BOS> {text} <EOS>"
        
        # El tokenizador ya nos devuelve la lista de IDs directamente
        ids = tokenizer.encode(text)
        tokenized_corpus.append(ids)
        
    return tokenized_corpus

print("Procesando corpus de entrenamiento con BPE.")
# Reutilizamos la variable `plaintext_corpora` de la celda de arriba
subword_corpora = []
for title, corpus in plaintext_corpora.items():
    subword_corpora.extend(subword_preprocess_corpus(corpus.sents()))

V_SUBWORD = len(tokenizer)
print(f"Nuevo tamaño total del vocabulario BPE (V_SUBWORD): {V_SUBWORD}")

# %% [markdown]
# ## 5. Evaluación de Vocabulario (Modelo Subwords)

# %%
# Procesamos Génesis con subwords
test_corpus_subword = subword_preprocess_corpus(genesis.sents())
# %%
# Contamos los tokens
test_subwords = [token_id for sent in test_corpus_subword for token_id in sent]
total_test_subwords = len(test_subwords)
# %% [markdown]
# ### OOV con sub‑palabras
#
# Verificamos que en el corpus de prueba casi no aparecen tokens desconocidos gracias al BPE.

# %%

# En un BPE pre-entrenado robusto, el OOV es prácticamente nulo porque 
# las palabras raras se parten en fragmentos conocidos. Aún así lo medimos:
train_vocab_subword = set(tokenizer.get_vocab().values())
oov_subwords = sum(1 for token_id in test_subwords if token_id not in train_vocab_subword)
oov_rate_subword = oov_subwords / total_test_subwords

print("Evaluación de Vocabulario (Modelo Subword)")
print(f"Tamaño del vocabulario subword: {V_SUBWORD}")
print(f"Tokens OOV en test: {oov_subwords}")
print(f"OOV Rate: {oov_rate_subword:.4%} ({oov_rate_subword:.4f})")

# %% [markdown]
# ## 6. Entrenamiento y Perplejidad del Modelo Subword
# Creamos trigramas de IDs BPE y entrenamos un nuevo modelo TrigramModel sobre ese vocabulario extendido.
#

# %%
import time
import numpy as np

def get_train_test_data_subword(corpus_ids, n=3):
    x_data = []
    y_data = []
    for sent in corpus_ids:
        if len(sent) < n: continue
        for i in range(len(sent) - n + 1):
            w1, w2, w3 = sent[i], sent[i+1], sent[i+2]
            x_data.append([w1, w2])
            y_data.append([w3])
    return x_data, y_data

print("Generando trigramas de subwords")
x_train_sub, y_train_sub = get_train_test_data_subword(subword_corpora, n=3)


BATCH_SIZE = 256

train_set_sub = np.concatenate((x_train_sub, y_train_sub), axis=1)
train_loader_sub = DataLoader(train_set_sub, batch_size=BATCH_SIZE)

print(f"Iniciando entrenamiento en {device}.")
model_subword = TrigramModel(V_SUBWORD, EMBEDDING_DIM, CONTEXT_SIZE, H).to(device)
optimizer_sub = optim.Adam(model_subword.parameters(), lr=2e-3)
loss_function_sub = nn.NLLLoss()

subword_model_path = os.path.join(NN_MODELS_PATH, "lm_subword_epoch_0.dat")

if os.path.exists(subword_model_path):
    print(f"\nModelo ya entrenado detectado en {subword_model_path}")
    model_subword.load_state_dict(torch.load(subword_model_path, map_location=device))
else:
    print("\nNo se encontró ningún modelo. Iniciando entrenamiento.")
    EPOCHS = 1
    for epoch in range(EPOCHS):
        st = time.time()
        for it, data_tensor in enumerate(train_loader_sub):
            context_tensor = data_tensor[:, 0:2].to(device)
            target_tensor = data_tensor[:, 2].to(device)

            model_subword.zero_grad()
            log_probs = model_subword(context_tensor)
            loss = loss_function_sub(log_probs, target_tensor)
            loss.backward()
            optimizer_sub.step()

            if it % 500 == 0:
                print(f"Iteración {it} | Loss: {loss.item():.4f} | Tiempo: {(time.time() - st):.2f}s")
                st = time.time()

    torch.save(model_subword.state_dict(), subword_model_path)
    print(f"\nModelo Subword guardado en: {subword_model_path}")
# %% [markdown]
# ### Evaluación de Perplejidad

# %%
print("\nEvaluando Perplejidad en Génesis.")
x_test_sub, y_test_sub = get_train_test_data_subword(test_corpus_subword, n=3)

x_tensor_sub = torch.tensor(x_test_sub)
y_tensor_sub = torch.tensor(y_test_sub).view(-1)

dataset_sub = TensorDataset(x_tensor_sub, y_tensor_sub)
dataloader_sub = DataLoader(dataset_sub, batch_size=512)

model_subword.eval()
loss_function_eval = nn.NLLLoss(reduction='sum')
total_loss_sub = 0.0
total_samples_sub = len(x_test_sub)

with torch.no_grad():
    for x_batch, y_batch in dataloader_sub:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        log_probs = model_subword(x_batch)
        loss = loss_function_eval(log_probs, y_batch)
        total_loss_sub += loss.item()

avg_loss_sub = total_loss_sub / total_samples_sub
ppl_subword = math.exp(avg_loss_sub)

print("Evaluación de Perplejidad (Modelo Subword)")
print(f"Pérdida (Cross-Entropy): {avg_loss_sub:.4f}")
print(f"Perplejidad: {ppl_subword:.4f}")

# %% [markdown]
# ## Extra

# %%
import random
import torch
import numpy as np

def generate_subword_sequence(model, prompt, max_tokens=30, top_p=0.8):
    model.eval()
    
    # 1. Preparamos el texto inicial y lo pasamos a IDs (Subwords)
    text = f"<BOS> {prompt}".lower()
    generated_ids = tokenizer.encode(text)
    
    # Necesitamos al menos 2 tokens por nuestro contexto (CONTEXT_SIZE=2)
    if len(generated_ids) < 2:
        print("Error: El prompt es muy corto. Intenta con al menos una palabra.")
        return ""
        
    with torch.no_grad():
        for _ in range(max_tokens):
            # 2. Tomamos los últimos 2 subwords como contexto
            context = generated_ids[-2:]
            context_tensor = torch.tensor([context]).to(device)
            
            # 3. El modelo escupe logaritmos de probabilidades
            log_probs = model(context_tensor).squeeze().tolist()
            
            # Juntamos (probabilidad real, ID) y ordenamos de mayor a menor
            probs_ids = [(np.exp(lp), idx) for idx, lp in enumerate(log_probs)]
            probs_ids.sort(key=lambda x: x[0], reverse=True)
            
            # 4. Aplicamos Top-P (Nucleus Sampling) para darle creatividad pero sentido
            cumulative_prob = 0.0
            valid_ids = []
            valid_probs = []
            
            for prob, idx in probs_ids:
                valid_ids.append(idx)
                valid_probs.append(prob)
                cumulative_prob += prob
                if cumulative_prob >= top_p:
                    break
                    
            # Elegimos el siguiente subword usando las probabilidades como peso
            next_id = random.choices(valid_ids, weights=valid_probs, k=1)[0]
            
            # 5. Agregamos a nuestra historia
            generated_ids.append(next_id)
            
            # Si el modelo decide que ya terminó la oración, paramos
            # Obtenemos el ID del <EOS> que metimos al tokenizer
            eos_id = tokenizer.encode("<EOS>")[-1]
            if next_id == eos_id:
                break
                
    # 6.Convertimos la lista de IDs de regreso a texto humano (palabras completas)
    final_text = tokenizer.decode(generated_ids)
    
    # Limpiamos los tokens especiales de inicio y fin para presentarlo bonito
    final_text = final_text.replace("<BOS>", "").replace("<EOS>", "").strip()
    return final_text

# %%
print("EJEMPLOS DE GENERACIÓN (MODELO SUBWORDS)\n")

prompts = [
    "god tells",
    "the people",
    "and then"
]

for i, p in enumerate(prompts):
    # Generamos usando nuestro modelo subword recién entrenado
    resultado = generate_subword_sequence(model_subword, p, max_tokens=25, top_p=0.85)
    print(f"Ejemplo {i+1} | Prompt: '{p}'")
    print(f"Generado: {resultado}\n")
