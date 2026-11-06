# Plataforma Web de Minería de Datos

Plataforma interactiva de Minería de Datos inspirada en Orange Data Mining, diseñada bajo una arquitectura en dos capas:
- **Capa de Procesamiento (`analisis/`):** Validación (RF1), limpieza automática (RF2), árboles de decisión (RF3), reglas de asociación (RF4) y generación de informes (RF5).
- **Capa de Presentación (`web/`):** Interfaz multipágina en Streamlit y paneles interactivos (RF6).

## Ejecución local
```bash
streamlit run web/app.py