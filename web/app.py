"""
Módulo: app.py
Capa: Presentación (web/)
Descripción: Punto de entrada principal de la plataforma web de Minería de Datos.
Inicializa el estado de la sesión y presenta el flujo de trabajo modular.
"""

import sys
from pathlib import Path
import streamlit as st

RUTA_RAIZ = Path(__file__).resolve().parent.parent
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))


def inicializar_estado_sesion() -> None:
    estados_por_defecto = {
        "df_original": None,
        "df_limpio": None,
        "metadata_archivo": None,
        "resumen_limpieza": None,
        "modelos_arbol": [],
        "reglas_asociacion": None,
        "config_asociacion": None
    }
    for clave, valor_inicial in estados_por_defecto.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor_inicial


def main() -> None:
    st.set_page_config(
        page_title="Plataforma de Minería de Datos",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inicializar_estado_sesion()

    st.title("Plataforma Web de Minería de Datos")
    st.markdown(
        """
        Bienvenido al entorno interactivo de **Minería de Datos**, diseñado bajo una arquitectura 
        modular de **dos capas** (*Procesamiento* y *Presentación*) e inspirado en el flujo visual de **Orange Data Mining**.
        """
    )

    st.divider()

    st.subheader("Estado Actual del Pipeline de Datos")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.session_state["df_original"] is not None:
            meta = st.session_state["metadata_archivo"]
            st.success(f"**1. Dataset Cargado**\n\n`{meta['nombre_archivo']}` ({meta['filas']} filas × {meta['columnas']} cols)")
        else:
            st.info("**1. Carga de Datos (RF1)**\n\nPendiente de cargar archivo CSV.")

    with col2:
        if st.session_state["df_limpio"] is not None:
            df_l = st.session_state["df_limpio"]
            st.success(f"**2. Datos Limpios (RF2)**\n\nListo ({df_l.shape[0]} filas × {df_l.shape[1]} cols)")
        else:
            st.warning("**2. Limpieza Automática (RF2)**\n\nPendiente de ejecutar limpieza.")

    with col3:
        n_modelos = len(st.session_state["modelos_arbol"])
        if n_modelos > 0:
            st.success(f"**3. Árboles de Decisión (RF3)**\n\n{n_modelos} modelo(s) guardado(s).")
        else:
            st.warning("**3. Árbol de Decisión (RF3)**\n\nSin modelos entrenados.")

    with col4:
        if st.session_state["reglas_asociacion"] is not None:
            n_reglas = len(st.session_state["reglas_asociacion"])
            st.success(f"**4. Asociación (RF4)**\n\n{n_reglas} regla(s) generada(s).")
        else:
            st.warning("**4. Reglas de Asociación (RF4)**\n\nSin reglas calculadas.")

    st.divider()

    st.subheader("Módulos del Sistema")
    st.markdown(
        """
        Utiliza el menú lateral izquierdo para recorrer secuencialmente los módulos de trabajo:
        1. ** Carga y Limpieza (RF1 y RF2):** Validación técnica de archivos CSV y preprocesamiento automático.
        2. ** Árbol de Decisión (RF3):** Entrenamiento supervisado, ajuste de hiperparámetros y gestión de modelos.
        3. ** Reglas de Asociación (RF4):** Descubrimiento de patrones frecuentes mediante algoritmos *Apriori* y *FP-Growth*.
        4. ** Dashboard e Informes (RF5 y RF6):** Visualización interactiva y exportación de reporte ejecutivo en HTML.
        """
    )

if __name__ == "__main__":
    main()