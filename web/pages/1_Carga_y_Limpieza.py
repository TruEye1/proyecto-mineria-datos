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
    # Hace accesibles los módulos de análisis al ejecutar la página desde Streamlit.
    sys.path.insert(0, str(RUTA_RAIZ))

from analisis.validador import validar_y_cargar_csv, TAMANO_MAXIMO_MB
from analisis.limpiador import limpiar_datos


def inicializar_estado_pagina() -> None:
    """Inicializa las claves de sesión usadas durante la carga y limpieza."""
    for clave in ["df_original", "df_limpio", "metadata_archivo", "resumen_limpieza"]:
        if clave not in st.session_state:
            st.session_state[clave] = None


def mostrar_diagnostico_archivo(metadata: dict, df: pd.DataFrame) -> None:
    """Muestra metadatos de carga y una vista previa del archivo original."""
    st.subheader(" Diagnóstico Técnico del Archivo (RF1)")
    # Resume dimensiones y características detectadas al validar el CSV.
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Filas", f"{metadata['filas']:,}")
    col2.metric("Columnas", f"{metadata['columnas']}")
    col3.metric("Codificación", metadata["encoding"].upper())
    col4.metric("Separador", metadata["nombre_separador"])
    col5.metric("Tamaño", f"{metadata['tamano_mb']:.4f} MB")

    # Permite inspeccionar algunas filas y tipos, nulos y proporción de faltantes.
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
    """Presenta los cambios de limpieza y una muestra del dataset resultante."""
    st.subheader("Resultados de la Limpieza Automática (RF2)")
    
    # Resume los principales cambios cuantitativos producidos por el proceso.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duplicados Eliminados", f"{resumen['duplicados_eliminados']}")
    c2.metric("Filas Finales", f"{resumen['filas_finales']}")
    c3.metric("Columnas Renombradas", "Sí" if resumen['columnas_renombradas'] else "No")
    total_atipicos = sum(resumen.get("atipicos_tratados", {}).values())
    c4.metric("Atípicos Tratados (IQR)", f"{total_atipicos}")

    # Detalla nulos, conversiones de tipo y valores atípicos tratados.
    if resumen["nulos_imputados"]:
        st.info(f"**Imputación de Nulos realizada en {len(resumen['nulos_imputados'])} columnas:**")
        for col, cant in resumen["nulos_imputados"].items():
            st.write(f"- `{col}`: {cant} valores imputados (Mediana para numéricos, Moda para categóricos).")
    else:
        st.success("No se encontraron valores nulos en el dataset.")
        
    if resumen["tipos_corregidos"]:
        st.info(f"**Tipos de datos corregidos automáticamente en:** {', '.join(resumen['tipos_corregidos'])}")

    if resumen.get("atipicos_tratados"):
        st.warning(f"**Valores atípicos neutralizados (Winsorización IQR) en {len(resumen['atipicos_tratados'])} columnas:**")
        for col, cant in resumen["atipicos_tratados"].items():
            st.write(f"- `{col}`: {cant} valores atípicos ajustados a los límites intercuartílicos.")

    # Muestra una muestra final para revisar el resultado de la limpieza.
    st.markdown("###  Vista Previa de Datos Limpios")
    st.dataframe(df_limpio.head(15), use_container_width=True)


def main() -> None:
    # Configura la interfaz y prepara el estado que persiste entre interacciones.
    st.set_page_config(page_title="Carga y Limpieza | Minería de Datos", page_icon="📂", layout="wide")
    inicializar_estado_pagina()

    st.title(" Módulo 1: Carga y Limpieza de Datos (RF1 y RF2)")
    
    # Recibe un CSV; la validación detallada se realiza en el módulo de análisis.
    archivo_subido = st.file_uploader(
        "Sube tu archivo CSV para comenzar:",
        type=["csv"],
        help=f"Máximo {TAMANO_MAXIMO_MB} MB. Detección automática de encoding y separador."
    )

    # Carga el archivo válido y descarta resultados anteriores si cambió el archivo.
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
            # Informa los errores de validación para que el usuario pueda corregir el archivo.
            st.error(f" {resultado['mensaje']}")
            for err in resultado["errores"]:
                st.warning(err)

    if st.session_state["df_original"] is not None:
        # Muestra el diagnóstico y ofrece ejecutar el pipeline de limpieza.
        mostrar_diagnostico_archivo(st.session_state["metadata_archivo"], st.session_state["df_original"])
        
        st.divider()

        st.subheader("Limpieza y Preprocesamiento Automático")
        st.markdown(
            """
            Al ejecutar la limpieza, el sistema:
            1. Estandarizará los nombres de las columnas (minúsculas, sin espacios).
            2. Eliminará registros duplicados.
            3. Corregirá tipos de datos mal formateados.
            4. Imputará valores nulos de forma inteligente (mediana/moda).
            5. Detectará y tratará valores atípicos mediante rango intercuartílico (IQR).
            """
        )

        # Ejecuta la limpieza bajo demanda y conserva datos y resumen en la sesión.
        if st.button(" Ejecutar Limpieza Automática", type="primary"):
            with st.spinner("Procesando y limpiando datos..."):
                df_limpio, resumen = limpiar_datos(st.session_state["df_original"])
                st.session_state["df_limpio"] = df_limpio
                st.session_state["resumen_limpieza"] = resumen
                
        if st.session_state["df_limpio"] is not None:
            mostrar_resumen_limpieza(st.session_state["resumen_limpieza"], st.session_state["df_limpio"])

if __name__ == "__main__":
    main()