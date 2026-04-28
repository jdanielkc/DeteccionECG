# Clasificación de Patologías Cardiovasculares con Máquinas de Aprendizaje Profundo a partir de Señales ECG

Trabajo de grado — Universidad del Valle, 2023  
**Autor:** José Daniel García Arias

---

## Descripción

Las enfermedades cardiovasculares representan una de las principales causas de muerte en Colombia y el mundo. Su detección temprana es crítica para reducir el riesgo de complicaciones graves. Este proyecto desarrolla un sistema de clasificación automática de patologías cardíacas a partir de señales ECG, combinando técnicas de procesamiento digital de señales con arquitecturas de aprendizaje profundo.

El sistema detecta cinco clases:

| Clase | Descripción |
|---|---|
| **Bradicardia** | Frecuencia cardíaca anormalmente baja |
| **Fibrilación Auricular (AFIB)** | Arritmia más común, asociada a riesgo de ACV |
| **Elevación del segmento ST (STE)** | Indicador de infarto agudo de miocardio |
| **Depresión del segmento ST (STD)** | Indicador de isquemia cardíaca |
| **Normal** | Ritmo sinusal sin alteraciones |

El mejor modelo (CNN con normalización estándar) alcanzó una **exactitud superior al 92%** en validación cruzada con pacientes no vistos durante el entrenamiento.

---

## Metodología

### 1. Bases de Datos

Se utilizaron dos bases de datos públicas disponibles en [PhysioNet](https://physionet.org):

- **MIT-BIH Arrhythmia Database**: 48 grabaciones de ECG de dos canales, muestreadas a 360 Hz, con ~110.000 anotaciones de latidos. Fuente de bradicardia y fibrilación auricular.
- **European ST-T Database**: 90 registros de ECG de 79 sujetos, muestreados a 250 Hz, con 367 episodios de cambio del segmento ST. Fuente de elevación y depresión del segmento ST.

### 2. Preprocesamiento de la Señal

Las señales pasan por una cadena de procesamiento secuencial:

1. **Filtro FIR** (pasa-alto, ventana Kaiser, 51 coeficientes, fc = 0.67 Hz): elimina el ruido de línea base.
2. **Filtro Notch** (IIR de segundo orden, 60 Hz): elimina interferencias de la red eléctrica.
3. **Interpolación y diezmado** (factor 25/36): homogeniza la frecuencia de muestreo de la base MIT-BIH de 360 Hz a 250 Hz.
4. **Normalización**: se evaluaron tres variantes — sin normalización, normalización de amplitud y normalización estándar. La normalización estándar produjo los mejores resultados.
5. **Enventanado y generación de espectrogramas**: ventanas de 10 segundos con 50% de solapamiento. Cada ventana genera un espectrograma de dimensiones (51, 241) usando `matplotlib.pyplot.specgram()` con NFFT=100 y noverlap=90. La dimensión final de cada muestra es **(51, 241, 2)** (dos canales).

Se generaron **6 bases de datos** en total (3 de entrenamiento y 3 de prueba), garantizando que los pacientes de prueba no estuvieran presentes en el entrenamiento para evitar la filtración de la firma cardíaca.

### 3. Arquitecturas Implementadas

#### Red Convolucional (CNN)

<img width="auto" height="auto" alt="Imagen32" src="https://github.com/user-attachments/assets/f9ba8771-868a-4385-b2ec-933560a8d530" />

- 2 capas convolucionales (filtros 3×3, activación ReLU) con max-pooling
- Capa de aplanamiento
- Capa densa con ReLU y dropout
- Capa de salida con softmax (5 clases)
- Entrada: espectrogramas de dimensión (51, 241, 2)

#### Red Recurrente (GRU)

<img width="auto" height="auto" alt="Imagen33" src="https://github.com/user-attachments/assets/1c12f5ff-c4d7-4b00-8a28-a546350f26ae" />


- 1 capa convolucional 1D (ReLU) + batch normalization
- 2 capas GRU con dropout
- Capa densa con regularización L1/L2
- Capa de salida con softmax (5 clases)
- Entrada: segmentos del espectrograma procesados secuencialmente

### 4. Optimización de Hiperparámetros

Se utilizó [Optuna](https://optuna.org) para la búsqueda automática de hiperparámetros, ejecutando **50 ciclos de entrenamiento de 30 épocas** por modelo. Los hiperparámetros optimizados incluyeron:

- **CNN**: dropout, número de neuronas en capa densa, tasa de aprendizaje (Adam)
- **GRU**: dropout, tamaño del kernel, strides, número de filtros, tasa de aprendizaje (Adam)

### 5. Resultados

| Modelo | Exactitud (Test) | F1-score | AUC |
|---|---|---|---|
| CNN + Normalización Estándar | **92.68%** | **92.27%** | **98.32%** |
| GRU + Normalización Estándar | 81.94% | 81.91% | 92.91% |

La CNN con normalización estándar fue el mejor modelo en todas las métricas. La GRU, con ~14 veces menos parámetros entrenables, logró resultados competitivos. La depresión del segmento ST (STD) fue la clase más difícil de clasificar en todos los modelos.

---

## Estructura del Repositorio

```
DeteccionECG/
├── EtapadeFiltrado/                  # Preprocesamiento de señales ECG
│   ├── EtapasFiltrado.ipynb          # Filtro FIR y Notch
│   ├── Normalizacion.ipynb           # Comparación de normalizaciones
│   ├── notch.ipynb                   # Análisis del filtro Notch
│   ├── Wavelets.ipynb                # Exploración con wavelets
│   └── GraficosBasesDatos.ipynb      # Visualización de las bases de datos
│
├── GeneracionBDConEspectrograma/     # Generación de bases de datos
│   ├── GeneracionBD_NA.ipynb         # BD con normalización de amplitud
│   ├── GeneracionBD_NS.ipynb         # BD con normalización estándar
│   └── GeneracionBD_SN.ipynb         # BD sin normalización
│
├── Arquitecturas/
│   ├── ArquitecturaConvolucional/    # Entrenamiento y evaluación CNN
│   │   ├── RedConvolucional_NA.ipynb
│   │   ├── RedConvolucional_NS.ipynb
│   │   └── RedConvolucional_SN.ipynb
│   ├── ArquitecturaRecurrente/       # Entrenamiento y evaluación GRU
│   │   ├── RedRecurrente_NA.ipynb
│   │   ├── RedRecurrente_NS.ipynb
│   │   └── RedRecurrente_SN.ipynb
│   └── ValidacionCruzada/            # Validación cruzada (5 pliegues)
│       ├── ArquitecturaConvolucional/
│       └── ArquitecturaRecurrente/
│
└── Cardiopatías___Deep_Networks/     # Documento de tesis (LaTeX + PDF)
```

Los sufijos en los notebooks indican el tipo de normalización:
- `_NA` → Normalización de Amplitud
- `_NS` → Normalización Estándar
- `_SN` → Sin Normalización

---

## Requisitos

- Python 3.7 o superior
- TensorFlow / Keras
- NumPy, SciPy, Pandas
- Scikit-learn
- Optuna
- Matplotlib, Seaborn
- WFDB (lectura de bases de datos PhysioNet)
- Jupyter Notebook

---

## Datos

Las bases de datos utilizadas son de acceso público en PhysioNet:

- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [European ST-T Database](https://physionet.org/content/edb/1.0.0/)

---

## Limitaciones

- Los modelos solo clasifican las cuatro patologías para las que fueron entrenados; no detectan otras afecciones cardíacas.
- El sistema no opera en tiempo real; está diseñado para el análisis de señales ECG previamente adquiridas y preprocesadas.
- No incluye interfaz de usuario.
- Los resultados deben validarse con bases de datos adicionales y mayor volumen de datos por clase.

---

## Trabajos Futuros

- Aplicar *data augmentation* para clases con pocas muestras (e.g., bradicardia).
- Explorar otras representaciones de la señal (wavelets, EMD, transformada coseno).
- Aumentar los ciclos de optimización y el rango de búsqueda de hiperparámetros.
- Desarrollar una interfaz de usuario para facilitar el uso del clasificador.
- Implementar el sistema en hardware embebido para clasificación en tiempo real.
