"""
Módulo: 4_Dashboard_e_Informes.py
Capa: Presentación (web/pages/)
Descripción: Panel interactivo con filtros dinámicos, distribuciones, correlaciones,
resultados de modelos (RF6) y generación/exportación de informe ejecutivo en HTML (RF5).
"""

import sys
from pathlib import Path
from typing import List
import pandas as pd
import streamlit as st
import plotly.express as px

RUTA_RAIZ = Path(__file__).resolve().parent.parent.parent
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from analisis.generador_informe import (
    calcular_estadisticas_dashboard,
    generar_conclusiones_automaticas,
    generar_html_informe,
)


def inicializar_estado() -> None:
    """Inicializa de forma segura las claves de sesión consumidas por el dashboard e informe."""
    valores_defecto = {
        "df_original": None,
        "df_limpio": None,
        "metadata_archivo": None,
        "resumen_limpieza": None,
        "modelos_arbol": [],
        "historial_asociacion": [],
        "reglas_asociacion": None,
        "config_asociacion": None,
    }
    for clave, valor in valores_defecto.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor


def aplicar_filtros_dinamicos(
    df: pd.DataFrame,
    cols_numericas: List[str],
    cols_categoricas: List[str],
) -> pd.DataFrame:
    """Renderiza controles de filtrado dinámico y retorna el DataFrame filtrado."""
    df_filtrado = df.copy()

    with st.expander("Filtros Dinámicos sobre los Datos", expanded=True):
        if cols_categoricas:
            st.markdown("**Filtros Categóricos**")
            cols_ui = st.columns(min(3, len(cols_categoricas)))
            for idx, col in enumerate(cols_categoricas):
                valores_unicos = sorted(df[col].dropna().astype(str).unique().tolist())
                with cols_ui[idx % len(cols_ui)]:
                    seleccion = st.multiselect(
                        f"{col}:",
                        options=valores_unicos,
                        default=valores_unicos,
                        key=f"filtro_cat_{col}",
                    )
                    if seleccion:
                        df_filtrado = df_filtrado[df_filtrado[col].astype(str).isin(seleccion)]
                    else:
                        df_filtrado = df_filtrado.iloc[0:0]

        if cols_numericas:
            st.markdown("**Filtros de Rango Numérico**")
            cols_num_ui = st.columns(min(3, len(cols_numericas)))
            for idx, col in enumerate(cols_numericas):
                serie = df[col].dropna()
                if serie.empty:
                    continue
                min_val = float(serie.min())
                max_val = float(serie.max())
                with cols_num_ui[idx % len(cols_num_ui)]:
                    if min_val < max_val:
                        rango = st.slider(
                            f"{col}:",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key=f"filtro_num_{col}",
                        )
                        df_filtrado = df_filtrado[
                            (df_filtrado[col] >= rango[0]) & (df_filtrado[col] <= rango[1])
                        ]
                    else:
                        st.caption(f"{col}: valor constante ({min_val:.2f})")

    return df_filtrado


def main() -> None:
    st.set_page_config(
        page_title="Dashboard e Informes | Minería de Datos",
        page_icon="📊",
        layout="wide",
    )
    inicializar_estado()

    st.title("Dashboard Interactivo e Informes (RF5 y RF6)")
    st.markdown(
        """
        Explora visualmente los datos limpios mediante **filtros dinámicos**, analiza distribuciones,
        correlaciones y rendimiento de modelos, y descarga el **informe ejecutivo en HTML**.
        """
    )

    if st.session_state["df_limpio"] is None:
        st.warning("**Prerrequisito no cumplido:** Debes cargar y limpiar un dataset primero.")
        st.info("Ve al módulo **1. Carga y Limpieza** en el menú lateral para subir y procesar tu archivo.")
        if st.session_state.get("df_original") is not None:
            st.info("Ya subiste un archivo, pero aún no presionas **Ejecutar Limpieza Automática**.")
        return

    df_limpio = st.session_state["df_limpio"]
    stats_base = calcular_estadisticas_dashboard(df_limpio)

    df_filtrado = aplicar_filtros_dinamicos(
        df_limpio,
        cols_numericas=stats_base["cols_numericas"],
        cols_categoricas=stats_base["cols_categoricas"],
    )
    stats = calcular_estadisticas_dashboard(df_filtrado)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registros Activos", f"{stats['filas']:,}", delta=f"de {stats_base['filas']:,} totales")
    k2.metric("Columnas", f"{stats['columnas']}")
    k3.metric("Variables Numéricas", f"{len(stats['cols_numericas'])}")
    k4.metric("Variables Categóricas", f"{len(stats['cols_categoricas'])}")

    st.divider()

    tab_dist, tab_corr, tab_modelos, tab_informe = st.tabs([
        "Distribuciones",
        "Correlaciones",
        "Resultados de Modelos",
        "Informe Ejecutivo (HTML)",
    ])

    with tab_dist:
        if df_filtrado.empty:
            st.warning("Ningún registro coincide con los filtros seleccionados. Ajusta los filtros para ver gráficos.")
        else:
            c_num, c_cat = st.columns(2)
            with c_num:
                if stats["cols_numericas"]:
                    col_num_sel = st.selectbox(
                        "Variable numérica a visualizar:",
                        options=stats["cols_numericas"],
                        key="dash_col_num",
                    )
                    fig_hist = px.histogram(
                        df_filtrado,
                        x=col_num_sel,
                        nbins=20,
                        marginal="box",
                        title=f"<b>Distribución de {col_num_sel}</b>",
                        color_discrete_sequence=["#2563eb"],
                    )
                    fig_hist.update_layout(height=400, margin=dict(t=50, l=40, r=30, b=40))
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.info("No hay variables numéricas disponibles para graficar.")

            with c_cat:
                if stats["cols_categoricas"]:
                    col_cat_sel = st.selectbox(
                        "Variable categórica a visualizar:",
                        options=stats["cols_categoricas"],
                        key="dash_col_cat",
                    )
                    conteo = (
                        df_filtrado[col_cat_sel]
                        .astype(str)
                        .value_counts()
                        .reset_index()
                    )
                    conteo.columns = [col_cat_sel, "Frecuencia"]
                    fig_bar = px.bar(
                        conteo,
                        x=col_cat_sel,
                        y="Frecuencia",
                        text_auto=True,
                        color="Frecuencia",
                        color_continuous_scale="Blues",
                        title=f"<b>Frecuencia de {col_cat_sel}</b>",
                    )
                    fig_bar.update_layout(height=400, margin=dict(t=50, l=40, r=30, b=40))
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No hay variables categóricas disponibles para graficar.")

            if not stats["resumen_numerico"].empty:
                with st.expander("Ver Estadísticas Descriptivas (Datos Filtrados)"):
                    st.dataframe(stats["resumen_numerico"], use_container_width=True)

    with tab_corr:
        if df_filtrado.empty:
            st.warning("Ningún registro coincide con los filtros seleccionados.")
        elif len(df_filtrado) < 2:
            st.info("Se necesitan al menos 2 registros filtrados para calcular correlaciones.")
        elif stats["matriz_correlacion"].empty:
            st.info("Se requieren al menos 2 columnas numéricas para calcular la matriz de correlación.")
        else:
            fig_corr = px.imshow(
                stats["matriz_correlacion"],
                text_auto=".2f",
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title="<b>Matriz de Correlación de Pearson</b>",
            )
            fig_corr.update_layout(height=450, margin=dict(t=50, l=60, r=40, b=50))
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab_modelos:
        modelos_arbol = st.session_state["modelos_arbol"]
        reglas_asoc = st.session_state["reglas_asociacion"]
        historial_asoc = st.session_state["historial_asociacion"]

        st.markdown("#### Modelos de Árbol de Decisión (RF3)")
        if modelos_arbol:
            df_mod = pd.DataFrame(modelos_arbol)
            df_melt = df_mod.melt(
                id_vars=["nombre"],
                value_vars=["exactitud", "precision", "recall", "f1_score"],
                var_name="Métrica",
                value_name="Valor",
            )
            df_melt["Porcentaje"] = (df_melt["Valor"] * 100).round(2)
            fig_mod = px.bar(
                df_melt,
                x="nombre",
                y="Porcentaje",
                color="Métrica",
                barmode="group",
                text_auto=".1f",
                title="<b>Comparativa de Modelos Guardados (%)</b>",
            )
            fig_mod.update_layout(height=380, yaxis_range=[0, 105])
            st.plotly_chart(fig_mod, use_container_width=True)
        else:
            st.info("Aún no has guardado modelos en el módulo **Árbol de Decisión**.")

        st.divider()
        st.markdown("#### Reglas de Asociación (RF4)")
        if reglas_asoc is not None and not reglas_asoc.empty:
            top_reglas = reglas_asoc.sort_values(by="lift", ascending=False).head(10)
            st.dataframe(
                top_reglas[["regla", "support", "confidence", "lift"]].rename(columns={
                    "regla": "Regla",
                    "support": "Soporte",
                    "confidence": "Confianza",
                    "lift": "Lift",
                }),
                use_container_width=True,
                hide_index=True,
            )
        elif historial_asoc:
            st.info(f"Tienes {len(historial_asoc)} análisis guardados en el historial de asociación.")
        else:
            st.info("Aún no se han generado reglas en el módulo **Reglas de Asociación**.")

    with tab_informe:
        st.markdown("#### Conclusiones Automáticas del Análisis")
        conclusiones = generar_conclusiones_automaticas(
            df=df_limpio,
            resumen_limpieza=st.session_state["resumen_limpieza"],
            modelos_arbol=st.session_state["modelos_arbol"],
            historial_asociacion=st.session_state["historial_asociacion"],
            reglas_activas=st.session_state["reglas_asociacion"],
        )
        for item in conclusiones:
            st.write(f"- {item}")

        st.divider()
        html_informe = generar_html_informe(
            metadata_archivo=st.session_state["metadata_archivo"],
            df_limpio=df_limpio,
            resumen_limpieza=st.session_state["resumen_limpieza"],
            modelos_arbol=st.session_state["modelos_arbol"],
            historial_asociacion=st.session_state["historial_asociacion"],
            reglas_activas=st.session_state["reglas_asociacion"],
        )
        st.download_button(
            label="Descargar Informe Ejecutivo (HTML)",
            data=html_informe.encode("utf-8"),
            file_name="informe_mineria_datos.html",
            mime="text/html",
            type="primary",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
