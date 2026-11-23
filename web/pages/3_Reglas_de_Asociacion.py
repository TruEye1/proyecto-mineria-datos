"""
Módulo: Reglas_de_Asociacion.py
Capa: Presentación (web/pages/)
Descripción: Interfaz interactiva para el descubrimiento de itemsets frecuentes
y reglas de asociación mediante Apriori y FP-Growth (RF4).
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px

# Resolver la raíz del proyecto de forma segura (funcione en web/pages/ o web/)
RUTA_ACTUAL = Path(__file__).resolve().parent
RUTA_RAIZ = RUTA_ACTUAL.parent.parent if RUTA_ACTUAL.name == "pages" else RUTA_ACTUAL.parent
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from analisis.asociacion import (
    preparar_transacciones,
    generar_reglas_asociacion
)


def inicializar_estado():
    if "df_limpio" not in st.session_state:
        st.session_state["df_limpio"] = None
    if "reglas_actual" not in st.session_state:
        st.session_state["reglas_actual"] = None
    if "historial_asociacion" not in st.session_state:
        st.session_state["historial_asociacion"] = []


def render_grafico_dispersion(reglas_df: pd.DataFrame):
    """Gráfico Soporte vs. Confianza coloreado por Lift."""
    df_plot = reglas_df.copy()
    df_plot["lift_size"] = df_plot["lift"].clip(lower=0.1)

    fig = px.scatter(
        df_plot,
        x="support",
        y="confidence",
        size="lift_size",
        color="lift",
        color_continuous_scale="Viridis",
        hover_name="regla",
        hover_data={
            "support": ":.3f",
            "confidence": ":.3f",
            "lift": ":.3f",
            "lift_size": False
        },
        labels={
            "support": "Soporte (Frecuencia)",
            "confidence": "Confianza (Probabilidad Condicional)",
            "lift": "Lift (Fuerza de Asociación)"
        },
        title="<b>Mapa de Reglas: Soporte vs. Confianza</b>"
    )
    fig.update_layout(height=420, margin=dict(t=50, l=50, r=40, b=50))
    return fig


def render_top_itemsets(df_itemsets: pd.DataFrame, top_n: int = 10):
    """Gráfico de barras horizontales con los itemsets más frecuentes."""
    df_top = df_itemsets.head(top_n).sort_values(by="support", ascending=True)
    fig = px.bar(
        df_top,
        x="support",
        y="items",
        orientation="h",
        color="support",
        color_continuous_scale="Blues",
        labels={"support": "Soporte", "items": "Conjunto de Ítems"},
        title=f"<b>Top {len(df_top)} Itemsets Frecuentes</b>"
    )
    fig.update_layout(height=380, margin=dict(t=45, l=120, r=40, b=40))
    return fig


def main():
    st.set_page_config(
        page_title="Reglas de Asociación | Minería de Datos",
        page_icon="🔗",
        layout="wide"
    )
    inicializar_estado()

    st.title(" Módulo 4: Reglas de Asociación (RF4)")
    st.markdown(
        """
        Este módulo descubre patrones frecuentes del tipo **Si ocurre A ➔ entonces ocurre B**
        utilizando los algoritmos **Apriori** y **FP-Growth** (`mlxtend`).
        """
    )

    # 1. Validación de prerrequisito
    if st.session_state["df_limpio"] is None:
        st.warning(" **Prerrequisito no cumplido:** Debes cargar y limpiar un dataset primero.")
        st.info(" Ve al módulo **1. Carga y Limpieza** en el menú lateral para subir y limpiar tu archivo.")
        if st.session_state.get("df_original") is not None:
            st.info(" Ya subiste un archivo, pero aún no presionas **Ejecutar Limpieza Automática**.")
        return

    df = st.session_state["df_limpio"]
    st.success(f" **Dataset activo:** {df.shape[0]} filas × {df.shape[1]} columnas disponibles.")

    st.divider()

    # 2. Selección de variables
    st.subheader("1. Configuración de Transacciones")
    columnas_disponibles = list(df.columns)
    default_cols = [c for c in columnas_disponibles if "id" not in c.lower()] or columnas_disponibles

    col_sel, col_opt = st.columns([2, 1])
    with col_sel:
        columnas_sel = st.multiselect(
            " Columnas a incluir en el análisis:",
            options=columnas_disponibles,
            default=default_cols,
            help="Se recomienda excluir identificadores únicos como id_cliente."
        )

    with col_opt:
        incluir_numericas = st.checkbox(
            "Incluir variables numéricas discretizadas",
            value=False,
            help="Agrupa columnas numéricas (ej. edad, ingreso) en rangos para poder asociarlas."
        )
        n_rangos = 3
        if incluir_numericas:
            n_rangos = st.slider("Cantidad de rangos (bins):", min_value=2, max_value=5, value=3, step=1)

    if not columnas_sel:
        st.error("Debes seleccionar al menos una columna para generar reglas.")
        return

    st.divider()

    # 3. Parámetros del algoritmo
    st.subheader("2. Parámetros del Algoritmo")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        algoritmo = st.selectbox(
            "Algoritmo:",
            options=["apriori", "fpgrowth"],
            format_func=lambda x: "Apriori" if x == "apriori" else "FP-Growth",
            help="Ambos encuentran los mismos itemsets; FP-Growth suele ser más rápido en datasets grandes."
        )

    with c2:
        min_support = st.slider(
            "Soporte Mínimo:",
            min_value=0.05,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Frecuencia mínima con la que debe aparecer una combinación en el dataset."
        )

    with c3:
        metrica = st.selectbox(
            "Métrica de Filtrado:",
            options=["confidence", "lift"],
            format_func=lambda x: "Confianza (Confidence)" if x == "confidence" else "Lift",
            help="Confianza mide probabilidad condicional; Lift > 1 indica asociación positiva real."
        )

    with c4:
        max_umbral = 1.0 if metrica == "confidence" else 5.0
        valor_defecto = 0.6 if metrica == "confidence" else 1.0
        min_threshold = st.slider(
            f"Umbral mínimo ({metrica.title()}):",
            min_value=0.1,
            max_value=max_umbral,
            value=valor_defecto,
            step=0.05
        )

    st.divider()

    # 4. Ejecución
    if st.button("⚡ Generar Reglas de Asociación", type="primary", use_container_width=True):
        st.session_state["reglas_actual"] = None

        with st.spinner("Construyendo matriz transaccional y extrayendo reglas..."):
            try:
                matriz_tx = preparar_transacciones(
                    df=df,
                    columnas_seleccionadas=columnas_sel,
                    incluir_numericas=incluir_numericas,
                    n_rangos=n_rangos
                )
                resultado = generar_reglas_asociacion(
                    df_transacciones=matriz_tx,
                    algoritmo=algoritmo,
                    min_support=min_support,
                    metrica=metrica,
                    min_threshold=min_threshold
                )

                st.session_state["reglas_actual"] = {
                    "resultado": resultado,
                    "matriz_tx": matriz_tx,
                    "parametros": {
                        "algoritmo": algoritmo,
                        "min_support": min_support,
                        "metrica": metrica,
                        "min_threshold": min_threshold,
                        "incluir_numericas": incluir_numericas
                    },
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                st.success("🎉 ¡Análisis de asociación completado con éxito!")

            except Exception as e:
                st.error(f" Error durante el análisis: {str(e)}")

    # 5. Resultados
    if st.session_state["reglas_actual"] is not None:
        activo = st.session_state["reglas_actual"]
        res = activo["resultado"]
        params = activo["parametros"]
        reglas_df = res["reglas"]

        st.subheader("3. Resumen de Resultados")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transacciones", res["total_transacciones"])
        m2.metric("Itemsets Frecuentes", res["total_itemsets"])
        m3.metric("Reglas Generadas", res["total_reglas"])
        m4.metric("Lift Máximo", f"{res['mejor_lift']:.2f}")

        st.divider()

        tab_reglas, tab_itemsets, tab_grafico, tab_guardar = st.tabs([
            " Reglas Encontradas",
            " Itemsets Frecuentes",
            " Soporte vs. Confianza",
            " Guardar Ejecución"
        ])

        with tab_reglas:
            if reglas_df.empty:
                st.info("No se encontraron reglas con estos parámetros. Intenta bajar el soporte o el umbral mínimo.")
            else:
                filtro_texto = st.text_input(
                    " Filtrar reglas por palabra clave (ej. Electronica, Iquique, Si):",
                    value=""
                ).strip()

                df_ver = reglas_df.copy()
                if filtro_texto:
                    df_ver = df_ver[df_ver["regla"].str.contains(filtro_texto, case=False, na=False)]

                cols_tabla = ["regla", "support", "confidence", "lift"]
                df_mostrar = df_ver[cols_tabla].rename(columns={
                    "regla": "Regla (Antecedente ➔ Consecuente)",
                    "support": "Soporte",
                    "confidence": "Confianza",
                    "lift": "Lift"
                })
                st.dataframe(
                    df_mostrar.style.format({
                        "Soporte": "{:.2%}",
                        "Confianza": "{:.2%}",
                        "Lift": "{:.2f}"
                    }),
                    use_container_width=True
                )

        with tab_itemsets:
            if res["itemsets"].empty:
                st.info("No se encontraron itemsets frecuentes para el soporte mínimo seleccionado.")
            else:
                fig_items = render_top_itemsets(res["itemsets"])
                st.plotly_chart(fig_items, use_container_width=True)
                st.dataframe(
                    res["itemsets"].rename(columns={
                        "items": "Conjunto de Ítems",
                        "longitud": "N° Ítems",
                        "support": "Soporte"
                    }),
                    use_container_width=True
                )

        with tab_grafico:
            if reglas_df.empty:
                st.info("No hay reglas suficientes para graficar.")
            else:
                fig_disp = render_grafico_dispersion(reglas_df)
                st.plotly_chart(fig_disp, use_container_width=True)

        with tab_guardar:
            nombre_def = f"{params['algoritmo'].upper()} (Sup {params['min_support']}) [{activo['timestamp']}]"
            c_nom, c_btn = st.columns([2, 1])
            with c_nom:
                nombre_run = st.text_input("Nombre del análisis:", value=nombre_def)
            with c_btn:
                st.write("")
                st.write("")
                if st.button("Guardar en Sesión", type="primary"):
                    st.session_state["historial_asociacion"].append({
                        "id": len(st.session_state["historial_asociacion"]) + 1,
                        "nombre": nombre_run,
                        "algoritmo": params["algoritmo"].upper(),
                        "min_support": params["min_support"],
                        "metrica": params["metrica"],
                        "umbral": params["min_threshold"],
                        "itemsets": res["total_itemsets"],
                        "reglas": res["total_reglas"],
                        "mejor_lift": round(res["mejor_lift"], 2),
                        "hora": activo["timestamp"]
                    })
                    st.success(f"✅ Análisis '{nombre_run}' guardado en la sesión.")

    # 6. Historial guardado
    st.divider()
    st.subheader("📋 Análisis Guardados en la Sesión")
    if st.session_state["historial_asociacion"]:
        st.dataframe(pd.DataFrame(st.session_state["historial_asociacion"]), use_container_width=True)
        if st.button("🗑️ Borrar Historial de Asociación"):
            st.session_state["historial_asociacion"] = []
            st.rerun()
    else:
        st.info("Aún no has guardado ningún análisis de reglas en esta sesión.")


if __name__ == "__main__":
    main()