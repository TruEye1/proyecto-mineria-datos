"""
Módulo: 1_Carga_y_Limpieza.py
Capa: Presentación (web/pages/)
Descripción: Interfaz gráfica para la carga (RF1) y limpieza automática (RF2) de datos.
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

RUTA_RAIZ = Path(__file__).resolve().parent.parent.parent
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from analisis.validador import validar_y_cargar_csv, TAMANO_MAXIMO_MB
from analisis.limpiador import limpiar_datos


def inicializar_estado_pagina() -> None:
    for clave in ["df_original", "df_limpio", "metadata_archivo", "resumen_limpieza"]:
        if clave not in st.session_state:
            st.session_state[clave] = None


def mostrar_diagnostico_archivo(metadata: dict, df: pd.DataFrame) -> None:
    st.subheader(" Diagnóstico Técnico del Archivo (RF1)")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Filas", f"{metadata['filas']:,}")
    col2.metric("Columnas", f"{metadata['columnas']}")
    col3.metric("Codificación", metadata["encoding"].upper())
    col4.metric("Separador", metadata["nombre_separador"])
    col5.metric("Tamaño", f"{metadata['tamano_mb']:.4f} MB")

    with st.expander(" Ver Datos Originales y Estructura", expanded=False):
        st.dataframe(df.head(15), use_container_width=True)
        
        resumen_cols = pd.DataFrame({
            "Columna": df.columns,
            "Tipo": [str(dtype) for dtype in df.dtypes],
            "Nulos": df.isnull().sum().values,
            "% Nulos": (df.isnull().sum().values / len(df) * 100).round(2)
        })
        st.dataframe(resumen_cols, use_container_width=True, hide_index=True)


def mostrar_resumen_limpieza(resumen: dict, df_limpio: pd.DataFrame) -> None:
    st.subheader("✨ Resultados de la Limpieza Automática (RF2)")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Duplicados Eliminados", f"{resumen['duplicados_eliminados']}")
    c2.metric("Filas Finales", f"{resumen['filas_finales']}")
    c3.metric("Columnas Renombradas", "Sí" if resumen['columnas_renombradas'] else "No")

    if resumen["nulos_imputados"]:
        st.info(f"**Imputación de Nulos realizada en {len(resumen['nulos_imputados'])} columnas:**")
        for col, cant in resumen["nulos_imputados"].items():
            st.write(f"- `{col}`: {cant} valores imputados (Mediana para numéricos, Moda para categóricos).")
    else:
        st.success("No se encontraron valores nulos en el dataset.")
        
    if resumen["tipos_corregidos"]:
        st.info(f"**Tipos de datos corregidos automáticamente en:** {', '.join(resumen['tipos_corregidos'])}")

    st.markdown("###  Vista Previa de Datos Limpios")
    st.dataframe(df_limpio.head(15), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Carga y Limpieza | Minería de Datos", page_icon="📂", layout="wide")
    inicializar_estado_pagina()

    st.title(" Módulo 1: Carga y Limpieza de Datos (RF1 y RF2)")
    
    archivo_subido = st.file_uploader(
        "Sube tu archivo CSV para comenzar:",
        type=["csv"],
        help=f"Máximo {TAMANO_MAXIMO_MB} MB. Detección automática de encoding y separador."
    )

    if archivo_subido is not None:
        resultado = validar_y_cargar_csv(archivo_subido)
        if resultado["valido"]:
            archivo_previo = (
                st.session_state["metadata_archivo"]["nombre_archivo"]
                if st.session_state["metadata_archivo"] else None
            )
            if archivo_previo != resultado["metadata"]["nombre_archivo"]:
                st.session_state["df_limpio"] = None
                st.session_state["resumen_limpieza"] = None

            st.session_state["df_original"] = resultado["dataframe"]
            st.session_state["metadata_archivo"] = resultado["metadata"]
        else:
            st.error(f" {resultado['mensaje']}")
            for err in resultado["errores"]:
                st.warning(err)

    if st.session_state["df_original"] is not None:
        mostrar_diagnostico_archivo(st.session_state["metadata_archivo"], st.session_state["df_original"])
        
        st.divider()

        st.subheader("🧹 Limpieza y Preprocesamiento Automático")
        st.markdown(
            """
            Al ejecutar la limpieza, el sistema:
            1. Estandarizará los nombres de las columnas (minúsculas, sin espacios).
            2. Eliminará registros duplicados.
            3. Corregirá tipos de datos mal formateados.
            4. Imputará valores nulos de forma inteligente.
            """
        )

        if st.button(" Ejecutar Limpieza Automática", type="primary"):
            with st.spinner("Procesando y limpiando datos..."):
                df_limpio, resumen = limpiar_datos(st.session_state["df_original"])
                st.session_state["df_limpio"] = df_limpio
                st.session_state["resumen_limpieza"] = resumen
                
        if st.session_state["df_limpio"] is not None:
            mostrar_resumen_limpieza(st.session_state["resumen_limpieza"], st.session_state["df_limpio"])

if __name__ == "__main__":
    main()