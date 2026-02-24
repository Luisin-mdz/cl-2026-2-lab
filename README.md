# Laboratorio de Lingüística Computacional 2026-2

Repositorio con las prácticas de laboratorio para la materia de Lingüística Computacional 2026-2

## Objetivo del lab

- Profundizar en uso de herramientas y desarrollo de soluciones a tareas de
  *Natural Language Processing (NLP)* enfocandonos en la reflexión desde una
  perspectiva lingüística (computacional)
- Ser unæ **destacadæ** practicante, tanto a nivel académico como industrial,
  del *NLP*
- Practicar lo que vean en clase de teoría :)

<center><img src="http://i0.kym-cdn.com/entries/icons/facebook/000/008/342/ihave.jpg"></center>

## Entregas

- **Entregables serán a través de GitHub usando git, forks y pull requests**
  - Para mas información revisa el notebook `notebooks/0_lab_intro.ipynb`
- Es muy recomendable entregar las prácticas ya que representa un porcentaje importante de su calificación (`30%`) 🤓
- Se dará ~2 semanas para entregar ejercicios (dependiendo de la práctica)
    - En caso de **entregas tardías** abrá una penalización `-1 punto` por cada día
    - Si la entrega sobre pasa 5 días la calificación máxima será sobre 6
- Si utilizas LLMs, o herramientas generativas reportalos en tus prácticas 🧙🏼‍♀️
  - Reporta el nivel de uso (*no judgement zone*):
    - 1: Corrección de estilo
    - 2: Estructura e ideas
    - 3: Co-autoría con agradecimientos en la tesis a shatcito
    - 4: Delegación supervisada
    - 5: Fuí expectador
> Les recomendamos ampliamente que lo intenten por su cuenta primero, es una oportunidad de enfrentarse a cosas nuevas y de pensar en soluciones nunca antes vistas :)

## Práctica 0: Crear un PR hacia el repositorio principal del laboratorio

- El PR deberá crear una carpeta con su username de GitHub dentro de `practicas/` y otra carpeta interna llamada `P0/`
    - `practicas/umoqnier/P0`
- Agrega un archivo llamado `README.md` a la carpeta `P0/` con información básica sobre tí y que esperas aprender en el lab. Ejemplo:
    - `practicas/umoqnier/P0/README.md`
    - Usar lenguaje de marcado [Markdown](https://docs.github.com/es/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

```markdown
$ cat README.md

# Diego Alberto Barriga Martínez

- Número de cuenta: `XXXXXXXX`
- User de Github: @umoqnier
- Me gusta que me llamen: Dieguito

## Expectativas

- Crear un shatjipiti
- Hacerle la competencia a openia
- Ganar millones de picafresas en un día con mi emprendimiento

## Pasatiempos

- Andar en bici

## Proyectos en los que he participado y que me enorgullesen 🖤

- [Esquite](https://github.com/ElotlMX/Esquite/)
```

## Tarea 1: Exploración de Niveles del lenguaje 🔭

### FECHA DE ENTREGA: 10 de Marzo 2026 at 11:59pm

### Fonética

1. Con base en el sistema de búsqueda visto en la práctica 1, dónde se recibe una palabra ortográfica y devuelve sus transcripciones fonológicas, proponga una solución para los casos en que la palabra buscada no se encuentra en el lexicón/diccionario.
    - ¿Cómo devolver o **aproximar** su transcripción fonológica?
    - Reutiliza el sistema de búsqueda visto en clase y mejóralo con esta funcionalidad.
    - Muestra al menos tres ejemplos

### Morfología

2. Elige tres lenguas del corpus que pertenezcan a familias lingüísticas distintas
   - Ejemplo: `spa` (Romance), `eng` (Germánica), `hun` (Urálica)
   - Para cada una de las tres lenguas calcula y compara:
       -  **Ratio morfemas / palabra**: El promedio de morfemas que componen las palabras
        -  **Indicé de Flexión / Derivación**: Del total de morfemas, ¿Qué porcentaje son etiquetas de flexión (`100`) y cuáles de derivación (`010`)?
3. Visualización
    - Genera una figura con **subplots** para comparar las lenguas lado a lado.
    - *Plot 1*: Distribución de la longitud de los morfemas
    - *Plot 2*: Distribución de las categorías (flexión, derivación, raíz, etc.)
4. Con base en esta información, responde la pregunta: *¿Cuál de las tres lenguas se comporta más como una lengua aglutinante y cuál como una lengua aislante?*
    - Justifica tu respuesta usando tus métricas y figuras

### EXTRA:

- Genera la [matriz de confusión](https://en.wikipedia.org/wiki/Confusion_matrix) para el etiquetador CRFs visto en clase
- Observando las etiquetas donde el modelo falló responde las preguntas:
    - ¿Por qué crees que se confundió?
    - ¿Es un problema de ambigüedad léxica (la palabra tiene múltiples etiquetas)?
    - ¿Qué *features* añadirías para solucionarlo?