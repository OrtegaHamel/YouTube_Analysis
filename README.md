# Simulador y Optimizador de Tracción para YouTube - v2.0.0

## Descripción
Este proyecto tiene como objetivo realizar un análisis exploratorio de datos (EDA) sobre distintos conjuntos de datos para identificar sus características, calidad y desafíos. A partir de este análisis, se selecciona un conjunto de datos y una problemática específica para desarrollar en etapas posteriores, justificando su relevancia y potencial de aplicación en técnicas de ciencia de datos y aprendizaje automático. Finalmente, se busca realizar el preprocesamiento de datos y la optimización de modelos de machine learning para el conjunto de datos seleccionado. La meta es elegir la técnica de machine learning más adecuada y optimizar sus hiperparámetros para obtener el mejor rendimiento posible.

# Análisis Inicial y Selección de Problema

## Conjuntos de Datos Analizados
Descripción breve de los cuatro conjuntos de datos analizados.

- **YouTube Video Statistics – Trending & Engagement** *(1_youtube_trending_video_statistics_5000.csv)*: 5.000 registros correspondientes a videos que alcanzaron a calificar como tendencias, distribuidos en diversas categorías.

- **YouTube Analytics Data** *(2_youtube_recommendation_dataset -.csv)*: 537 registros donde cada fila consolida las características técnicas y los indicadores clave de rendimiento (KPIs) del video.

- **YouTube data for Analytics** *(3_youtube_data.csv)* Colección de metadatos de 600 videos de YouTube. Contiene información detallada para profundizar en el mundo de las tendencias de contenido digital.

- **YouTube Videos and Channels Metadata** *(4_YouTubeDataset_withChannelElapsed.csv)*: Metadatos de videos y canales de YouTube para analizar la relación estadística entre las publicaciones, con foco en el comportamiento del consumidor.

## Resumen del EDA Inicial

- **YouTube Video Statistics – Trending & Engagement**: A partir del EDA inicial, se sospecha que el Data 1 corresponde a un conjunto de datos sintético o artificialmente controlado. Se presenta estructuralmente limpio (con 0 registros duplicados y 0 valores nulos), pero el análisis de la matriz de correlación reveló que todas las variables numéricas (views, likes, comments, duration_seconds y engagement_rate) presentan una independencia estadística absoluta entre sí, mostrando coeficientes de correlación prácticamente nulos. La ruptura de esta lógica en el dataset confirma su naturaleza artificial o preprocesada, aislando el comportamiento de las métricas numéricas. 

- **YouTube Analytics Data**: Estructuralmente presenta un estado limpio de un conjunto de datos orgánicos y altamente especializados en el nicho de la tecnología. Sus distribuciones muestran asimetrías positivas severas y la presencia de outliers masivos, conviviendo contenidos breves con megatutoriales de programación de alta duración. La matriz de correlación de Spearman demostró interacciones orgánicas verdaderas, exhibiendo una fuerte sinergia lineal entre vistas y likes (0.83), un efecto de penalización por viralidad en las tasas de enganche y una redundancia matemática perfecta (1.00) entre el ratio de likes y la tasa de interacción.

- **YouTube data for Analytics**: Se observa un dataset contemporáneo de 600 registros concentrado en el nicho de viajes y tecnología, con un 3.1% de duplicados y un 22.6% de descripciones ausentes. Asimismo, exhibe fuertes asimetrías por videos virales atípicos, pero confirma interacciones orgánicas muy robustas mediante correlaciones de Spearman altamente significativas entre vistas, likes (0.854) y comentarios (0.743).

- **YouTube Videos and Channels Metadata**: El análisis exploratorio reveló un registro histórico de 10 años (2005-2015) con una dispersión extrema de 555K videos únicos, una pérdida crítica del 96.50% en la columna likes/dislikes y una fuerte asimetría por el fenómeno de la Cola Larga. La matriz de Spearman demostró una consistencia orgánica dominante (0.950) entre las vistas y la velocidad de maduración temporal.

## Problema Seleccionado

### Descripción del problema
En el ecosistema de la creación de contenido digital y el marketing de medios, se enfrentan a una incertidumbre crítica antes de publicar un video: la incapacidad de anticipar científicamente el rendimiento, la velocidad de penetración y el potencial de engagement de un lanzamiento. Esta carencia de herramientas analíticas cuantitativas pre-estreno provoca ineficiencias financieras severas, estancamiento de canales en el fenómeno de la "Cola Larga" (*Long Tail*) y la pérdida del costo de oportunidad de distribución en la plataforma. El problema central radica en transformar un volumen masivo de registros históricos de la plataforma en un motor capaz de modelar interacciones no lineales complejas para diagnosticar, predecir y prescribir los factores que determinan la velocidad de tracción de un contenido antes de que este sea subido de forma pública. Para ello, se selecciona el dataset **YouTube Videos and Channels Metadata.** *(YouTubeDataset_withChannelElapsed.csv)*

### Justificación
La pertinencia e innovación de este proyecto se sustentan sobre tres pilares analíticos y metodológicos:

1. **Rigor Histórico y Estructural:** El modelo se entrena utilizando una cápsula de tiempo arqueológica y masiva (Dataset 4: 575,506 registros sintonizados) que cubre de manera orgánica la primera década completa de vida de YouTube (2005-2015). Al analizar esta ventana, el algoritmo captura las dinámicas estructurales fundamentales de retención de audiencia y la maduración orgánica del algoritmo de recomendación original, aislando las predicciones de modas pasajeras o sesgos de estacionalidad contemporánea.
2. **Mitigación Exhaustiva de Fuga de Datos (*Data Leakage*):** A diferencia de los enfoques predictivos tradicionales que utilizan métricas posteriores al estreno (como conteo de vistas finales o me gusta en tiempo real), este sistema aísla rigurosamente las variables predictoras a la configuración exclusiva del creador pre-lanzamiento. Esto garantiza que la herramienta sea verdaderamente útil como un simulador predictivo.
3. **Resiliencia ante Datos Imperfectos:** El proyecto demuestra madurez en la Ciencia de Datos al reconfigurar el vector analítico de *engagement* mediante el ratio relativo de comentarios (`comments/views`), compensando de forma metodológicamente defendible una tasa de corrupción y censura crítica del 96.50% detectada en la variable `likes/dislikes`. La matriz de correlación de Spearman validó que esta reestructuración matemática posee una dinámica de activación comunitaria propia (correlación de 0.950 con el volumen de visualizaciones y estabilidad de 0.761 en tasas de participación), sustituyendo la señal perdida y aportando un valor predictivo de largo plazo para la toma de decisiones comerciales.

### Objetivos específicos

* **Diseñar un Pipeline de Preprocesamiento Automatizado (`ColumnTransformer`):** Construir un flujo de ingeniería de características robusto y reproducible en Python que descomponga estampas temporales complejas mediante el accesor `.dt` en variables cíclicas discretas (año, mes, día y hora) y aplique estrategias avanzadas de codificación no lineal (como *Target Encoding* supervisado o agregación volumétrica) para intermediar de forma eficiente la alta cardinalidad de creadores (`channelId` con casi medio millón de valores únicos) sin colapsar la memoria del sistema.
* **Implementar y Calibrar Modelos de Ensamble basados en Árboles No Paramétricos:** Desarrollar y evaluar algoritmos avanzados como *Random Forest*, *XGBoost* o *LightGBM* capaces de capturar interacciones complejas no lineales entre la duración, temática y temporalidad, optimizando sus hiperparámetros (`max_depth`, `min_samples_leaf`) mediante `GridSearchCV` con validación cruzada ($cv=5$) para mitigar activamente el sobreajuste provocado por el desbalance intrínseco de la muestra.
* **Desarrollar el Módulo Predictivo de Simulación en Tiempo Real:** Configurar una interfaz de software (El Predictor) que consuma el pipeline entrenado para procesar las variables de configuración del creador antes del estreno y devuelva una estimación cuantitativa del Índice de Velocidad de Tracción Esperado (`views/elapsedtime`), minimizando la desviación del MAE (Error Absoluto Medio) frente al comportamiento real observacional.
* **Construir el Módulo Prescriptivo de Recomendación Estratégica:** Estructurar un motor de reglas automatizado (El Optimizador) basado en la extracción de fronteras de decisión y Gráficos de Dependencia Parcial (PDP) que le permita al sistema prescribir científicamente los "puntos dulces" empíricos de lanzamiento (la combinación ideal de día, hora y duración) para maximizar el retorno de inversión (ROI) en una categoría temática específica.  

## Resumen de Resultados

### 1. Importancia de Variables (Enfoque de Negocio)
El modelo final basado en **LightGBM** identificó las siguientes características clave mediante el análisis de Ganancia (*Gain*):

| Posición | Variable | Importancia por GAIN (Poder Predictivo) | Importancia por SPLIT (Frecuencia) |
| :---: | :--- | :---: | :---: |
| 1 | `channelId` | 610,381,664.0 | 102 |
| 2 | `videoCount` | 385,028,518.0 | 222 |
| 3 | `subscriberCount` | 124,061,355.0 | 98 |
| 4 | `channelViewCount` | 110,462,514.0 | 108 |
| 5 | `publish_hour` | **91,989,048.0** | **100** |

*Nota: La alta relevancia de `publish_hour` valida matemáticamente la viabilidad del Módulo Optimizador para recetar ventanas óptimas de lanzamiento.*

### 2. Comparativa Evolutiva de Modelos
A través del ciclo de desarrollo se lograron los siguientes hitos de precisión en el conjunto de prueba independiente:

| Módulo del Proceso | MAE en Set de Test | R² en Set de Test | Estado y Progreso |
| :--- | :---: | :---: | :--- |
| **Modelo Inicial (Base)** | 2.4636 | 0.0147 | Desempeño por defecto inicial. |
| **GridSearchCV Exhaustivo** | 2.1281 | -0.0739 | Ajuste por grilla discreta rígida. |
| **Optuna (Optimización Bayesiana)** | **1.9727** | **0.0107** | **Ganador Definitivo (Reducción del ~20% del error)** |

## Instrucciones de Ejecución

Para reproducir el análisis de este proyecto, sigue estos pasos:

### Configuración del Entorno  
Clona este repositorio en tu equipo local.  

### Instalación de Dependencias  
Instala las librerías necesarias ejecutando el siguiente comando en tu terminal:
   ```bash
   pip install -r requirements.txt
   ```

### Preparación de Datos
* **Nota Importante:** Debido a las limitaciones de tamaño en el repositorio, el dataset `4_YouTubeDataset_withChannelElapsed.csv` no se encuentra incluido.
* **Acción requerida:** Descarga el archivo y ubícalo dentro de la carpeta `/data/` para asegurar que el código pueda localizarlo correctamente.

### Ejecución
1. **Entorno**: Abre los *Notebooks* de Jupyter (`.ipynb`) utilizando tu entorno de desarrollo configurado.
2. **Proceso**: Ejecuta las celdas en orden secuencial para cargar los datos, procesarlos y visualizar los hallazgos del análisis.

## Autor
Álvaro Ortega Hamel — Desarrollo, análisis y documentación del proyecto.

## Licencia
MIT license