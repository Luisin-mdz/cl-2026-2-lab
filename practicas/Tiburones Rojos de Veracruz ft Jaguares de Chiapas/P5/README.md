# P5: Fine-tuning de ALBERT para análisis de sentimientos

En esta práctica hacemos fine-tuning a un modelo preentrenado para realizar análisis de sentimientos.

La tarea consiste en tomar una oración en inglés y clasificar si tiene connotación positiva o negativa.

---

## Contenido

```text
.
├── P5.py                        # Código fuente 
├── P5.ipynb                     # Notebook generado
├── app.py                       # Prototipo en Gradio
├── requirements.txt             # Dependencias para Hugging Face Spaces
├── README.md                    # Este README
└── modelo-sentimientos/         # Modelo fine-tuneado
```

---

## Requisitos

- `torch`
- `transformers`
- `datasets`
- `gradio`
- `sentencepiece`
- `safetensors`
- `codecarbon`

---

## Tarea seleccionada

Se eligió **análisis de sentimientos** como tarea NLP relevante. El modelo toma una oración en inglés y clasifica si tiene connotación positiva o negativa.

Ejemplos:

```text
"This movie was really good." → positivo
"This class was boring and terrible." → negativo
```

---

## Dataset

Utilizamos el dataset **SST-2**, incluido en GLUE.

La estructura del dataset es la siguiente:

- `sentence`: oración de entrada.
- `label`: etiqueta de sentimiento.
- `idx`: identificador del ejemplo.

Las etiquetas son:

- `0`: negativo
- `1`: positivo

Se utilizó un subconjunto del dataset con:

- 3000 ejemplos para entrenamiento.
- 500 ejemplos de validación.

---

## Modelo base

Se utilizó el modelo preentrenado **ALBERT**:

```text
albert/albert-base-v2
```

ALBERT es un transformer entrenado con un corpus en inglés de forma autosupervisada. En esta práctica se utilizó como base para una tarea de clasificación de secuencias, agregando una cabeza de clasificación binaria para distinguir entre sentimientos positivos y negativos.

---

## Fine-tuning

El entrenamiento se realizó con `Trainer` de Hugging Face.

Configuración utilizada:

```text
Épocas: 1
Batch size de entrenamiento: 8
Batch size de evaluación: 8
Ejemplos de entrenamiento: 3000
Ejemplos de validación: 500
Hardware: CPU
```

---

## Desempeño

Al haber problemas con la librería `evaluate` y conflictos relacionados con `torchvision`, se implementó **accuracy** manualmente a partir de los logits del modelo.

```python
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}
```

### Resultados obtenidos

```text
eval_loss: 0.5353
eval_accuracy: 0.732
```

Con estos resultados, el modelo logra clasificar correctamente una parte considerable de las oraciones usadas en los ejemplos de validación. Sin embargo, el desempeño debe interpretarse tomando en cuenta que el entrenamiento fue limitado a un subconjunto del dataset.

---

## Prototipo

Se desarrolló una aplicación con **Gradio** para probar el modelo desde una interfaz web.

La app permite escribir una oración en inglés y devuelve las probabilidades asociadas a las clases:

- `negativo`
- `positivo`

## URL pública del prototipo:

```text
https://huggingface.co/spaces/luisin0/analisis-sentimientos-ALBERT
```

---

## Extra: CodeCarbon

Para el punto extra se integró **CodeCarbon** en la aplicación.

La app reporta una estimación de emisiones por predicción en kg CO₂ equivalente. Como cada inferencia es muy rápida, los valores obtenidos suelen ser pequeños, pero permiten documentar el costo computacional aproximado del prototipo.

---

## Retos y dificultades

1. **Configuración en PyTorch**

   `uv` instalaba la versión CUDA de PyTorch. Al no contar con una GPU NVIDIA dedicada, se optó por forzar la utilización de PyTorch para CPU en el `pyproject.toml`.

   ```toml
   [tool.uv.sources]
   torch = { index = "pytorch-cpu" }

   [[tool.uv.index]]
   name = "pytorch-cpu"
   url = "https://download.pytorch.org/whl/cpu"
   explicit = true
   ```

2. **Evaluación**

   La librería `evaluate` produjo conflictos relacionados con `torchvision`, por lo que se optó por implementar la métrica de forma manual.

3. **Tiempo de entrenamiento**

   El entrenamiento completo sobre SST-2 era demasiado lento en CPU, así que se trabajó con un subconjunto de 3000 ejemplos para entrenamiento y 500 para validación.

---

## Limitaciones

1. **Entrenamiento**

   El entrenamiento fue limitado a 3000 oraciones y la validación a 500 oraciones durante una sola época.

2. **Lenguaje**

   El corpus utilizado sólo contiene ejemplos en inglés, por lo que el modelo no debe utilizarse para clasificar oraciones en español.

3. **Ambigüedad**

   El modelo puede fallar con oraciones ambiguas, irónicas o con sentimientos mixtos.

---

## Uso de IA

Se utilizaron herramientas de IA como apoyo en:

1. Depuración del entorno local.
2. Revisión de este README.

El código final fue revisado, ejecutado y adaptado manualmente para esta práctica.

---

## Autores

Luisin-mdz 
SubsetOfMars
