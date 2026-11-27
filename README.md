# Plataforma Web de Minería de Datos

Plataforma interactiva de Minería de Datos inspirada en **Orange Data Mining**, diseñada bajo una arquitectura modular de **dos capas** (*Procesamiento y Análisis* en Python y *Presentación Web* en Streamlit).

## Arquitectura del Proyecto

```text
proyecto-mineria-datos/
├── analisis/                   # Capa de procesamiento y análisis (Python puro)
│   ├── validador.py            # RF1: Validación de CSV (extensión, tamaño, encoding, separador)
│   ├── limpiador.py            # RF2: Limpieza automática (columnas, duplicados, tipos, nulos, IQR)
│   ├── arbol_decision.py       # RF3: Preparación, entrenamiento, métricas y reglas de árbol
│   ├── asociacion.py           # RF4: Matriz transaccional y reglas (Apriori / FP-Growth)
│   └── generador_informe.py    # RF5/RF6: Estadísticas de dashboard, conclusiones e informe HTML
├── web/                        # Capa de presentación (Streamlit + Plotly)
│   ├── app.py                  # Punto de entrada y resumen del estado del pipeline
│   └── pages/
│       ├── 1_Carga_y_Limpieza.py
│       ├── 2_Arbol_de_Decision.py
│       ├── 3_Reglas_de_Asociacion.py
│       └── 4_Dashboard_e_Informes.py
├── tests/
│   ├── dataset_prueba.csv      # Dataset de prueba con duplicados, nulos y valores atípicos
│   └── test_pipeline.py        # Script de verificación técnica del pipeline
└── requirements.txt
```

## Requisitos e Instalación

- Python 3.10 o superior.
- Instalar dependencias desde la raíz del repositorio:

```powershell
python -m pip install -r requirements.txt
```

## Ejecución de la Aplicación Web

Inicia la plataforma desde la raíz del proyecto:

```powershell
python -m streamlit run web/app.py
```

### Flujo de Uso por Módulos
1. **Carga y Limpieza (RF1 y RF2):** Sube un archivo `.csv` (hasta 50 MB), revisa el diagnóstico técnico automático y ejecuta la limpieza automática.
2. **Árbol de Decisión (RF3):** Elige variable objetivo y predictoras, ajusta hiperparámetros (`criterion`, `max_depth`, `min_samples_split`, `test_size`), entrena, guarda, compara o elimina modelos.
3. **Reglas de Asociación (RF4):** Configura columnas, discretización opcional, algoritmo (`Apriori` o `FP-Growth`), soporte y confianza/lift; filtra, ordena, guarda y recarga análisis.
4. **Dashboard e Informes (RF5 y RF6):** Aplica filtros dinámicos sobre los datos limpios, explora distribuciones, correlaciones y comparativas de modelos, y descarga el **Informe Ejecutivo en HTML**.

## Verificación y Pruebas

Para ejecutar las pruebas automatizadas del pipeline sobre `tests/dataset_prueba.csv`:

```powershell
python tests/test_pipeline.py
```
