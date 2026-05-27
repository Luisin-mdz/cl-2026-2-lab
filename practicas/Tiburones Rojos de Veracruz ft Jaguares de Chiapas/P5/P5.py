# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Práctica 5: Fine-tuning de ALBERT para análisis de sentimientos
#
# En esta práctica se realiza fine-tuning de un modelo transformer preentrenado para clasificar oraciones como positivas o negativas.

# %% [markdown]
# ## 1. Imports

# %%
import torch
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments


# %% [markdown]
# ## 2. Configuración general

# %%
MODEL_CHECKPOINT = "albert/albert-base-v2"
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "sst2"

OUTPUT_DIR = "sentiment-trainer"
FINAL_MODEL_DIR = "modelo-sentimientos"

SEED = 42
TRAIN_SIZE = 3000
EVAL_SIZE = 500


# %% [markdown]
# ## 3. Carga del dataset

# %%
raw_datasets = load_dataset(DATASET_NAME, DATASET_CONFIG)

# %%
raw_datasets

# %%
raw_datasets["train"][0]

# %%
raw_datasets["train"][9105]


# %% [markdown]
# ## 4. Tokenizador

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# %%
tokenizer("This cheese is good.")


# %% [markdown]
# ## 5. Tokenización del dataset

# %%
def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)


# %%
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# %%
tokenized_datasets["train"][0]


# %% [markdown]
# ## 6. Padding dinámico

# %%
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# %% [markdown]
# ## 7. Carga del modelo

# %%
id2label = {
    0: "negativo",
    1: "positivo",
}

label2id = {
    "negativo": 0,
    "positivo": 1,
}

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)


# %% [markdown]
# ## 8. Métrica de evaluación
# Definimos manualmente la métrica de accuracy. Esta métrica mide la proporción de ejemplos clasificados correctamente. Se implementó directamente con NumPy para evitar depender de la librería `evaluate`, que causó conflictos con `torchvision` en el entorno local.

# %%
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()

    return {
        "accuracy": accuracy,
    }


# %% [markdown]
# ## 9. Subconjuntos de entrenamiento y validación

# %%
small_train_dataset = (
    tokenized_datasets["train"]
    .shuffle(seed=SEED)
    .select(range(TRAIN_SIZE))
)

small_eval_dataset = (
    tokenized_datasets["validation"]
    .shuffle(seed=SEED)
    .select(range(EVAL_SIZE))
)

# %%
small_train_dataset[0]

# %%
small_eval_dataset[0]


# %% [markdown]
# ## 10. Configuración del entrenamiento

# %%
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    report_to="none",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    dataloader_pin_memory=False,
)


# %% [markdown]
# ## 11. Entrenamiento

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %%
train_results = trainer.train()

# %%
train_results


# %% [markdown]
# ## 12. Evaluación

# %%
from transformers.utils.notebook import NotebookProgressCallback

trainer.remove_callback(NotebookProgressCallback)
eval_results = trainer.evaluate()

# %%
eval_results


# %% [markdown]
# ## 13. Guardado del modelo

# %%
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)


# %% [markdown]
# ## 14. Predicción manual

# %%
def predict_sentiment(text):
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]

    negative_score = probs[0].item()
    positive_score = probs[1].item()

    if positive_score > negative_score:
        predicted_label = "positivo"
        confidence = positive_score
    else:
        predicted_label = "negativo"
        confidence = negative_score

    return {
        "texto": text,
        "predicción": predicted_label,
        "confianza": confidence,
        "score_negativo": negative_score,
        "score_positivo": positive_score,
    }


# %%
predict_sentiment("This movie was really good.")

# %%
predict_sentiment("This class was boring and terrible.")

# %%
predict_sentiment("The food was okay, but the service was awful.")

# %%
