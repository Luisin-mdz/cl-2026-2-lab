# P4: Evaluación de modelos del lenguaje neuronales

## Evaluando la perplejidad de un modelo de lenguaje
La perplejidad de un modelo de lenguaje es el índice de certeza que experimenta un modelo cuando predice el siguiente token de una cadena. Mide qué tan "sorprendido" está el modelo con dicha predicción. Así, una perplejidad baja nos indica un modelo robusto, y una perplejidad alta nos indica que el entrenamiento deberá tener más intervención humana.

## Calculando la perplejidad
Para entender la perplejidad primero necesitamos dos conceptos, la entropía y la entropía cruzada.

### Entropía
La entropía mide la incertidumbre promedio de una distribución de probabilidad. Para un modelo de lenguaje, si $P$ es la distribución real de las palabras, la entropía se define como:

$$
H(P) = -\sum_{i} p(w_i) \log p(w_i)
$$

- $p(w_i)$ es la probabilidad real de que aparezca la $i$-ésima palabra en un contexto dado.
- La suma recorre todo el vocabulario.
- El signo negativo asegura que $H(P) \ge 0$ (porque $\log p(w_i)$ siempre es negativo o cero).

Cuanto mayor es $H(P)$, más impredecible es la secuencia de palabras; cuanto menor, más predecible.

### Entropía cruzada
Mientras la entropía solo mira la distribución real $P$, la entropía cruzada compara dos distribuciones: la verdadera $P$ y la que predice nuestro modelo, $Q$. Se define como:

$$
H(P,Q) = -\sum_{i} p(x_i) \log q(x_i)
$$

donde:
- $p(x_i)$ es la probabilidad real del resultado $x_i$ (la palabra correcta según los datos).
- $q(x_i)$ es la probabilidad que el modelo asigna a ese mismo resultado.
- La suma vuelve a recorrer todos los posibles resultados (todo el vocabulario).

Esta fórmula penaliza las predicciones incorrectas: si el modelo asigna una probabilidad baja ($q(x_i)$ pequeña) a la palabra real, $-\log q(x_i)$ se vuelve muy grande, aumentando la entropía cruzada. Por tanto, una entropía cruzada baja indica que el modelo está alineado con la distribución real; una alta, que el modelo está “sorprendido” y sus predicciones difieren de lo esperado.

### De la entropía cruzada a la perplejidad
La perplejidad no es más que la exponenciación de la entropía cruzada:

  $$
  \text{Perplejidad} = e^{H(P,Q)}
  $$

Esta transformación tiene una interpretación muy intuitiva: la perplejidad indica, en promedio, entre cuántas opciones igualmente probables está “eligiendo” el modelo para la siguiente palabra. Por ejemplo, una perplejidad de 10 significa que, en promedio, la incertidumbre del modelo equivale a escoger uniformemente entre 10 palabras posibles.

Así, una perplejidad baja refleja un modelo que concentra su probabilidad en pocas palabras (está más seguro de la siguiente palabra), mientras que una perplejidad alta muestra un modelo muy indeciso, que reparte su probabilidad entre muchas alternativas (está “más sorprendido”). Por eso decimos que una baja perplejidad indica un modelo más robusto, y una alta sugiere que el entrenamiento puede beneficiarse de más datos, ajuste de hiperparámetros o revisión de la arquitectura.
### Ventajas
- Es eficiente computacionalmente hablando.
- Es muy intuitiva.

### Desventajas
- Depende del tokenizador y del vocabulario.
- No es muy fiable en textos cortos.

---
### Tabla comparativa 
| Métrica               | Modelo Base | Modelo Subword |
|-----------------------|-------------|----------------|
| Perplejidad (genesis) |    397.17   |      9404      |
| Tamaño vocabulario    |    41392    |      50260     |
| OOV Rate              |    36.63%   |       0        | 

Aunque el modelo base tiene menor perplejidad (397), su elevado OOV (36 %) significa que falla por completo en más de un tercio del corpus. El modelo subword, con 0 % de OOV, es mucho más robusto frente a palabras nuevas, pero su perplejidad altísima indica que el entrenamiento fue insuficiente o que el vocabulario BPE usado es muy pequeño.

### Sugerencias
- Para el modelo base: Limpiar el corpus para disminuir el oov rate
- Para el modelo subword: aumentar epocas de entrenamiento
### Link al modelo
https://drive.google.com/drive/folders/1CZEEomS3YUiEtNn62j8Oc7ES75T2uxg7?usp=drive_link

## Contenido
```text
.
├── P4.py                        # Código fuente 
├── P4.ipynb                     # Notebook generado 
└── README.md
```
### Código para cargar el modelo en memoria
```python
import torch

# Parámetros del modelo Subword
V_SUBWORD = 50260 
EMBEDDING_DIM = 200
CONTEXT_SIZE = 2
H = 100
device = "cpu" # o "cuda"

model_subword = TrigramModel(V_SUBWORD, EMBEDDING_DIM, CONTEXT_SIZE, H).to(device)

# Cargar pesos
path_al_archivo = "ruta/a/tu/descarga/lm_subword_epoch_0.dat"
model_subword.load_state_dict(torch.load(path_al_archivo, map_location=device))
model_subword.eval()

print("Modelo cargado en memoria.")
```

## Uso de IA

Se utilizó **Gemini** como herramienta de apoyo en:

1. **Redacción de este README.**
2. Algunos bloques de Markdown en el notebook.
El resto del código (salvo partes explícitamente reutilizadas de clase) fue desarrollado de manera independiente.

---
## Fuentes consultadas para la perplejidad
https://www.comet.com/site/blog/perplexity-for-llm-evaluation/

### Nota técnica sobre dependencias de PyTorch (Error de CUDA)
Si al momento de ejecutar `import torch` el intérprete arroja un error de sistema relacionado con dependencias dinámicas (por ejemplo: `OSError: libcudart.so.13: cannot open shared object file` o `ValueError: libcublasLt.so.*[0-9] not found`), se debe a un conflicto con las librerías de CUDA precompiladas en el entorno virtual.

Para solucionar este inconveniente de forma rápida y poder evaluar la libreta, se recomienda instalar la versión de PyTorch exclusiva para CPU ejecutando el siguiente comando en su entorno:

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
## Autores

Luisin-mdz (Luis Alejandro Méndez Pérez) 
SubsetOfMars (Eunice Contreras Ortiz)
