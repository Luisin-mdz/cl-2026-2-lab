# Práctica 1: Niveles Lingüísticos – Fonética y Morfología

## Descripción

Este proyecto implementa dos módulos de análisis lingüístico computacional
como parte de la Práctica 1 del curso:

1. **Fonética:** Sistema de búsqueda de transcripciones fonológicas (IPA) con
   tolerancia a errores ortográficos mediante distancia de edición.
2. **Morfología:** Análisis comparativo de tres lenguas de familias lingüísticas
   distintas (húngaro, inglés y ruso) usando métricas morfológicas y
   visualización.

## Contenido

```
.
├── P1.py                       # Código fuente (formato Jupytext percent)
├── P1.ipynb                    # Notebook generado
└── README.md
```

## Requisitos

```
uv add requests pandas matplotlib numpy editdistance jupytext
```

## Parte 1: Fonética

Se reutiliza el sistema de búsqueda de la Práctica 1(clase), que recibe una palabra
ortográfica y devuelve su transcripción fonológica desde el diccionario
[IPA-dict](https://github.com/open-dict-data/ipa-dict).

### Mejora implementada

Cuando la palabra **no se encuentra** en el lexicón, el sistema busca la
palabra más cercana utilizando **distancia de edición (Levenshtein)** y
devuelve la transcripción de dicha palabra aproximada.

## Parte 2: Morfología

Se analizan tres lenguas de familias lingüísticas distintas usando el corpus
de [SIGMORPHON 2022 Shared Task](https://github.com/sigmorphon/2022SegmentationST):

| Lengua  | Familia   | Código |
|---------|-----------|--------|
| Húngaro | Urálica   | `hun`  |
| Inglés  | Germánica | `eng`  |
| Ruso    | Eslava    | `rus`  |

### Métricas calculadas

- **Ratio morfemas/palabra:** Promedio de morfemas por palabra.
- **Índice de flexión:** Porcentaje de palabras con categoría `100`.
- **Índice de derivación:** Porcentaje de palabras con categoría `010`.

### Visualización

Se genera una figura con dos subplots:

- **Plot 1:** Distribución de la longitud de los morfemas (boxplot).
- **Plot 2:** Distribución de categorías morfológicas — raíz, flexión,
  derivación, compuesto, y combinaciones (barras agrupadas con porcentajes).

## Uso de IA

Se utilizó **GitHub Copilot** como herramienta de apoyo en dos aspectos
específicos de este proyecto:

1. **Generación de la gráfica comparativa** (`matplotlib` subplots con
   boxplot y barras agrupadas).
2. **Redacción de este README.**
3. **Redacción de algunos bloques markdown en el notebook**
El resto del código (salvo código reutilizado) fue
desarrollado de manera independiente.

## Autor

Luisin-mdz
