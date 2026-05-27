# P5cod: Fine tunneo de ALBERT para análisis de sentimientos.

En esta práctica haremos fine tunning a un modelo preentrenado para realizar fine tunning para analizar sentimientos.
---

## Contenido

```text
.
├── P5.py                        # Código fuente 
├── P5.ipynb                     # Notebook generado 
└── README.md                    # Este Readme
```


---

## Requisitos

 - 'torch'
 - 'transformers'
 - 'datasets'
 - 'gradio'
 - 'sentencepiece'
 - 'safetensors'
 - 'codecarbon'

---



### Tarea seleccionada

Se eligió **análisis de sentimientos** como tarea NLP relevante, tomamos una oración en inglés y clasificamos si tiene connotación positiva o negativa.

---

## Dataset
Utilizamos el dataset **SST-2**, incluido en GLUE

la estructura del dataset e sla siguiente:

- 'sentence': oración de entrada.
- 'label': etiqueda de sentimiento.
- 'idx': identificador del ejemplo.

Las etiquetas son:
- 0 = negativo
- 1 = positivo

Se utilizó un subconjunto del dataset con: 

- 3000 ejemplos para entrenamiento
- 500 ejemplos de validación

---
## Modelo base

Se utilizó el modelo pre-entrenado: ALBERT

```text
albert/albert-base-v2
```
Un transformer entrenado con un corpus en inglés de forma auto-supervisada pensado principalmente para predecir la palabra faltante en una oración, recomendado por Huggingface para tarea de clasificación de secuencias.

## Desempeño
Al haber problemas con la libreria **Torchvision** se implementó **accuracy**, calculada manualmente a partir de los logits del modelo.

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

Por lo que el modelo logra clasificar correctamente los sentimientos de las oraciones usadas en los ejemplos de validación.

### Retos y dificultades.

1.**Configuración en PyTorch**
    - uv instala la versión CUDA de PyTorch, al no contar con una GPU Nvidia dedicada se optó por forzar la utilización de PyTorch para CPU's en el toml, si el proyecto será ejecutado en una computadora con GPU Nvidia no debería haber problema alguno, en caso contrario utilizar lo siguiente. 
    ```text
    [tool.uv.sources]
    torch = { index = "pytorch-cpu" }

    [[tool.uv.index]]
    name = "pytorch-cpu"
    url = "https://download.pytorch.org/whl/cpu"
    explicit = true
    ```
2. **Evaluación**
   - La libreria `evaluate` no pudo ser cambiada por conflictos de `torchvision`, se optó por implementar la métrica de forma manual.

## Limitaciones

1. **Entrenamiento**
    - El entrenamiento fue limitado a 3000 oraciones y la verificación a 500 oraciones por solo una epoca.
2. **Lenguaje**
   - El corpus utilizado sólo contiene ejemplos en inglés.
## Uso de IA

Se utilizó **GitHub Copilot** como herramienta de apoyo en:

1. **Redacción de este README.**

El resto del código (salvo partes explícitamente reutilizadas de clase) fue desarrollado de manera independiente.

---

## Autor

Luisin-mdz
