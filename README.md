# Plataforma Web de Minería de Datos

Aplicación interactiva para explorar datos y aplicar técnicas de minería de datos en un flujo guiado inspirado en Orange Data Mining.

Está organizada en dos capas:

- **Procesamiento (`analisis/`)**: validación, limpieza, árboles de decisión, reglas de asociación y generación de informes.
- **Presentación (`web/`)**: interfaz multipágina desarrollada con Streamlit y visualizaciones interactivas.

## Funcionalidades

### 1. Carga y limpieza de datos

- Importación de archivos CSV de hasta 50 MB.
- Detección automática de codificación, delimitador y tamaño.
- Diagnóstico de tipos de datos, valores nulos y estructura del archivo.
- Normalización de nombres de columnas.
- Eliminación de filas duplicadas.
- Conversión automática de columnas numéricas.
- Imputación de nulos con mediana o moda.
- Tratamiento de valores atípicos mediante el rango intercuartílico (IQR).
- Vista previa del dataset limpio.

### 2. Árbol de decisión

- Selección de variable objetivo y variables predictoras.
- Codificación one-hot de variables categóricas.
- División reproducible de los datos en entrenamiento y prueba.
- Configuración de criterio, profundidad máxima y tamaño de prueba.
- Métricas de exactitud, precisión, recall y F1-score.
- Matriz de confusión, reporte de clasificación e importancia de variables.
- Visualización gráfica del árbol y exportación de reglas IF-THEN.
- Comparación y eliminación de modelos guardados durante la sesión.

### 3. Reglas de asociación

- Preparación one-hot de variables categóricas.
- Discretización opcional de variables numéricas.
- Generación de itemsets con los algoritmos Apriori o FP-Growth.
- Filtrado por soporte, confianza y lift.
- Visualizaciones de soporte, confianza y lift.
- Historial de análisis dentro de la sesión.

### 4. Dashboard e informes

- Resumen consolidado del dataset y los resultados registrados.
- Historial y comparación gráfica de modelos y ejecuciones de asociación.
- Conclusiones automáticas generadas a partir de datos, limpieza, modelos y reglas.
- Vista previa y descarga del informe ejecutivo autocontenido en `.html`.
- Vista previa y descarga de un reporte de sesión en `.txt`.
- Descarga de las reglas activas en formato `.csv`.

Los datos y resultados se conservan únicamente en el estado de la sesión de Streamlit; al reiniciar la aplicación se pierde el estado actual.

## Tecnologías

- Python
- Streamlit
- pandas y NumPy
- scikit-learn
- mlxtend
- Plotly y Matplotlib

## Requisitos

- Python 3.9 o superior
- Entorno virtual recomendado

Las dependencias principales se encuentran en `requirements.txt`.

## Instalación y ejecución local

Desde la raíz del proyecto, crea y activa un entorno virtual:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloquea la activación del entorno, puedes ejecutarlo con:

```powershell
.venv\Scripts\activate.bat
```

Inicia la aplicación:

```powershell
streamlit run web/app.py
```

Streamlit mostrará una URL local, normalmente `http://localhost:8501`.

## Uso

1. Abre **Carga y Limpieza** y sube un archivo CSV.
2. Ejecuta la limpieza automática.
3. Continúa con **Árbol de Decisión** para entrenar y comparar clasificadores.
4. Usa **Reglas de Asociación** para explorar patrones frecuentes.
5. Consulta **Dashboard e Informes** para revisar y exportar los resultados.

Se incluye un dataset de ejemplo en `tests/dataset_prueba.csv`.

## Verificación

El script de pruebas comprueba el pipeline completo: carga CSV, limpieza, entrenamiento y evaluación del árbol, Apriori, FP-Growth y generación de informes (`.html`, `.txt` y `.csv`).

```powershell
python tests/test_pipeline.py
```

## Estructura del proyecto

```text
.
├── analisis/
│   ├── validador.py
│   ├── limpiador.py
│   ├── arbol_decision.py
│   ├── asociacion.py
│   └── generador_informe.py
├── web/
│   ├── app.py
│   └── pages/
│       ├── 1_Carga_y_Limpieza.py
│       ├── 2_Arbol_de_Decision.py
│       ├── 3_Reglas_de_Asociacion.py
│       └── 4_Dashboard_e_Informes.py
├── tests/
│   ├── dataset_prueba.csv
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```