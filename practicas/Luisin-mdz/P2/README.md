# P2: Ley de Zipf + Diversidad lingüística (México vs Perú)

Este proyecto explora (1) si un “idioma” artificial generado aleatoriamente cumple la **ley de Zipf**, (2) cómo se comporta un **corpus de pocos recursos** (Tzotzil) frente a Zipf, y (3) una visualización de **diversidad lingüística** mediante datos geográficos y genealogía de lenguas (Glottolog).


---

## Contenido

```text
.
├── P2.py                        # Código fuente 
├── P2.ipynb                     # Notebook generado 
├── biblia_tzotzil.pdf            # Corpus Tzotzil (ver sección "Datos")
└── data/
    ├── languages_and_dialects_geo.csv
    └── languoid.csv
```

> Nota: en el repositorio está **solo `P2.py` / `P2.ipynb`**, y los datos pesados descargarse desde Drive (ver “Datos”).

---

## Requisitos

Dependencias principales:

- `pdfplumber`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `plotly` 
- `jupytext`

Ejemplo de instalación (una opción):

```bash
pip install pdfplumber numpy pandas scipy matplotlib plotly jupytext
```

---

## 1) Procesamiento de texto (PDF)

El proyecto define funciones para:

- **Extraer texto desde un PDF** (`pdfplumber`)
- Normalizar:
  - minúsculas
  - remover dígitos
  - remover puntuación (manteniendo `'`)
- Tokenizar a lista de palabras
- Contar frecuencia con `Counter`
- Construir un `DataFrame` ordenado con:
  - `Palabras`
  - `Frecuencia`
  - `Rango` (rank)

Funciones principales:

- `limpia_pdf(pdf_path) -> list[str]`
- `contar_palabras(texto) -> Counter`
- `crear_dataframe(conteo) -> pd.DataFrame`

---

## 2) Idioma artificial: ¿sigue Zipf?

Se genera un “texto” artificial así:

1. Se define un alfabeto (incluye `ñ` y el espacio).
2. Se muestrean aleatoriamente ~5,000,000 caracteres.
3. Se unen en un string y se separa por espacios, produciendo “palabras”.
4. Se cuentan frecuencias y se grafica:
   - Frecuencia vs rango (escala lineal)
   - Frecuencia vs rango (escala log-log)

### Resultado esperado

El texto aleatorio **no** se ajusta bien a Zipf: la curva no se alinea con una recta en log-log y el parámetro `α` estimado no se aproxima a ~1 como suele ocurrir en lenguas naturales.

---

## 3) Estimación del parámetro α (Zipf)

Se estima `α` minimizando el error cuadrático en espacio logarítmico.

- Se toma `ranks = 1..N`
- Se toma `frequencies` ordenadas
- Se define la función objetivo:

\[
\log(f_r) \approx \log(f_1) - \alpha \log(r)
\]

Se usa:

- `scipy.optimize.minimize(...)`

Se reporta:

- `Estimated alpha`
- `Mean Squared Error`

Y se gráfica el ajuste:

- puntos reales en log-log
- curva ajustada `f_1 * r^{-α}`

---

## 4) Corpus con pocos recursos: Biblia en Tzotzil (Zinacantán)

Se analiza un PDF llamado `biblia_tzotzil.pdf`:

- Se limpia y tokeniza
- Se calcula frecuencia y ranking
- Se grafica Zipf (lineal y log-log)
- Se estima `α` igual que en el caso artificial

### Resultado esperado

Al ser una lengua natural, el corpus **sí** muestra un comportamiento cercano a Zipf: en log-log la distribución se aproxima más a una línea y el ajuste con `α` suele ser cercano a 1.

---

## 5) Diversidad lingüística: México vs Perú

Se reutiliza un dataset geográfico/genealógico (formato CSV):

- `data/languages_and_dialects_geo.csv`
- `data/languoid.csv`

### Flujo

1. Se filtran lenguas por “caja” geográfica (lat/long) para México y Perú.
2. Se reconstruye el **linaje genealógico** usando `languoid.csv`:
   - función `reconstruir_linaje(glottocode)`
   - sube por `parent_id` hasta la raíz
   - filtra nodos “bookkeeping” o `Unclassifiable`
3. Se crea la columna `Family` (primer nodo del árbol).
4. Se grafica con `plotly.express.scatter_geo`:
   - color por `Family`
   - hover con el nombre

### Comparación México vs Perú

Se compara:

- cantidad total de lenguas en cada región
- número de familias lingüísticas distintas (`nunique()`)

En el notebook se discute que México tiene más lenguas en términos absolutos, pero un número de familias comparable.

---

## Datos (importante)

Este proyecto requiere **dos recursos externos**:

1. **`biblia_tzotzil.pdf`**  
   - Debe estar en la raíz del proyecto (misma carpeta que `P2.py`)
   - Este archivo **lo subiré a Drive** para evitar cargas pesadas al repositorio 

2. **Carpeta `data/`** con los CSV:
   - `data/languages_and_dialects_geo.csv`
   - `data/languoid.csv`
   - En principio **deberías ya tener esta carpeta** (por material de clase), pero **también la subiré a Drive** para evitar inconsistencias de versión o pérdida de archivos.

Link al corpus en Tzotzil y a la carpeta data : https://drive.google.com/drive/folders/1XQSg9jwjeBdZtQNuIzSgh4OZO7G6UaOM?usp=sharing 

---

## Uso de IA

Se utilizó **GitHub Copilot** como herramienta de apoyo en:

1. **Redacción de este README.**

El resto del código (salvo partes explícitamente reutilizadas de clase) fue desarrollado de manera independiente.

---

## Autor

Luisin-mdz
