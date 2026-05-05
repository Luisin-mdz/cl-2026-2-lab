# Práctica 2: Propiedades Estadísticas del Lenguaje y Diversidad

Este directorio contiene los materiales correspondientes a la Práctica 2 del laboratorio de **Lingüística Computacional 2026-2**. En esta práctica se exploran las propiedades estadísticas del lenguaje (como la Ley de Zipf), la diversidad lingüística de diversas regiones y el uso de herramientas modernas para el Reconocimiento de Entidades Nombradas (NER).

## Archivos Principales

*   `p2_lang_stat.ipynb`: Notebook principal con el análisis de la Ley de Zipf para lenguajes artificiales y el idioma Otomí, mapas de diversidad lingüística (México vs. Rusia) y análisis de NER.
*   `p2_lang_stat.py`: Versión en script de Python generada a partir del notebook.
*   `data/`: Directorio que contiene los recursos necesarios para las visualizaciones geográficas y el análisis.

## Instrucciones de Ejecución

Para ejecutar los archivos de esta carpeta, asegúrate de seguir estos pasos:

1.  **Activar el entorno virtual**: Utiliza el entorno creado con `uv` en la raíz del proyecto.
    ```bash
    source ../../../.venv/bin/activate
    ```
2.  **Ejecutar Jupyter**: Abre el notebook `p2_lang_stat.ipynb` con Jupyter Lab o VS Code.
    ```bash
    jupyter lab p2_lang_stat.ipynb
    ```
3.  **Descarga de datasets**: El notebook incluye comandos `%pip install` para obtener dependencias adicionales dinámicamente si no están presentes.

## Dependencias Adicionales

Aunque el proyecto base utiliza un entorno gestionado por `uv`, para esta práctica se han utilizado dependencias adicionales que suelen instalarse de forma interactiva o manual dentro del notebook:

*   **elotl**: Utilizado para la carga de corpus de lenguas indígenas (Otomí).
*   **geopandas**: Necesario para el procesamiento de datos geoespaciales y la generación de mapas de diversidad.
*   **spacy**: Específicamente el modelo `es_core_news_sm` para el procesamiento de lenguaje natural y NER en español.
*   **datasets**: De Hugging Face, para cargar corpus externos de medicina, tweets y contratos.

Para instalar el modelo de spacy manualmente:
```bash
python -m spacy download es_core_news_sm
```

## Soporte de LLMs (Nivel 2)

Para el desarrollo de este notebook y la resolución de las tareas de la práctica, se integró el uso de Modelos de Lenguaje de Gran Escala (LLMs). En este **Nivel 2** del curso, el apoyo de estos modelos fue fundamental para:

*   **Generación de Código para Visualización**: Los LLMs asistieron en la creación de scripts complejos de `plotly` y `matplotlib` para visualizar las distribuciones de la Ley de Zipf y las frecuencias de etiquetas de NER.
*   **Implementación de Modelos Estadísticos**: Ayuda en la integración de `spaCy` y la lógica de extracción de entidades nombradas en múltiples dominios.
*   **Documentación y Comentarios**: Redacción de análisis lingüísticos y técnicos en español mexicano, facilitando la interpretación de los resultados estadísticos obtenidos.
*   **Optimización del Pipeline**: Ajuste de funciones para el procesamiento de datos provenientes de Hugging Face y otros repositorios externos.
