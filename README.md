# ECGDetection

Este repositorio contiene el código desarrollado para la tesis *"Clasificación de Patologías Cardiovasculares con Máquinas de Aprendizaje Profundo a partir de Señales ECG"*, realizado en la Universidad del Valle (2023) por José Daniel García Arias.

El objetivo del proyecto es implementar y optimizar dos arquitecturas de deep learning (CNN y RNN) para detectar, a partir de señales ECG, cuatro patologías cardíacas específicas:
- **Bradicardia**
- **Fibrilación Auricular**
- **Elevación del segmento ST**
- **Depresión del segmento ST**

La aproximación se basa en un completo preprocesamiento de la señal (filtrado FIR, Notch, interpolación, diezmado y normalización), seguido de la generación de espectrogramas para alimentar los modelos, y culmina con la optimización automática de hiperparámetros usando [Optuna](https://optuna.org).

---

## Estructura del Repositorio

El repositorio está organizado en tres módulos principales:

- **Arquitecturas**: Contiene las implementaciones de las redes neuronales (por ejemplo, modelos CNN y RNN) usadas para la clasificación.
- **EtapadeFiltrado**: Scripts y notebooks para el preprocesamiento de las señales ECG, incluyendo técnicas de filtrado y normalización.
- **GeneracionBDConEspectrograma**: Herramientas para la generación de bases de datos a partir de espectrogramas de señales ECG.

Además, se incluyen notebooks que permiten reproducir experimentos, evaluar el rendimiento de los modelos y visualizar los resultados.

---

## Requisitos

- **Python**: 3.7 o superior.
- NumPy, SciPy, Pandas
- Scikit-learn
- TensorFlow keras
- Optuna (para la optimización de hiperparámetros)
- Matplotlib y Seaborn (para la visualización)
- Jupyter Notebook (para la ejecución de los notebooks)
