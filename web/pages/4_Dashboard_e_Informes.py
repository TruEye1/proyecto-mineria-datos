"""
Módulo: 4_Dashboard_e_Informes.py
Capa: Presentación (web/pages/)
Descripción: Vista consolidada de resultados de la sesión y descarga de reportes.
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# Resolver la raíz del proyecto para importar módulos internos de forma consistente.
RUTA_ACTUAL = Path(__file__).resolve().parent
RUTA_RAIZ = RUTA_ACTUAL.parent.parent if RUTA_ACTUAL.name == "pages" else RUTA_ACTUAL.parent
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from analisis.generador_informe import (
    construir_resumen_sesion,
    generar_informe_texto,
    obtener_reglas_exportables,
)


def main():
    st.set_page_config(
        page_title="Dashboard e Informes | Minería de Datos",
        page_icon="📊",
        layout="wide",
    )

    st.title("Dashboard e Informes de Sesión")
    st.markdown(
        "Resumen consolidado del dataset activo, modelos entrenados y reglas de asociación "
        "registradas durante la sesión actual."
    )

    resumen = construir_resumen_sesion(st.session_state)

    # Métricas generales consolidadas a partir del estado de la sesión.
    st.subheader("Resumen General")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Filas Dataset Limpio",
        resumen["filas_limpias"] if resumen["tiene_dataset_limpio"] else "Sin datos",
    )
    c2.metric("Modelos de Árbol Guardados", resumen["total_modelos_arbol"])
    c3.metric("Análisis de Reglas Guardados", resumen["total_analisis_reglas"])
    mejor_lift_str = (
        f"{resumen['mejor_lift_reglas']:.2f}"
        if resumen["mejor_lift_reglas"] is not None
        else "N/A"
    )
    c4.metric("Mejor Lift Registrado", mejor_lift_str)

    st.divider()

    # Historiales y gráficos comparativos de los módulos supervisado y no supervisado.
    historial_arboles = st.session_state.get("modelos_arbol", [])
    historial_reglas = st.session_state.get("historial_asociacion", [])

    col_izq, col_der = st.columns(2)

    with col_izq:
        st.subheader("Historial de Árboles de Decisión")
        if historial_arboles:
            df_arboles = pd.DataFrame(historial_arboles)
            df_arboles_vista = df_arboles.copy()
            for col_met in ["exactitud", "precision", "recall", "f1_score"]:
                if col_met in df_arboles_vista.columns:
                    df_arboles_vista[col_met] = df_arboles_vista[col_met].apply(
                        lambda x: f"{float(x) * 100:.2f}%" if isinstance(x, (int, float)) else x
                    )
            st.dataframe(df_arboles_vista, use_container_width=True)

            if {"nombre", "exactitud"}.issubset(df_arboles.columns):
                df_acc_plot = df_arboles.copy()
                df_acc_plot["exactitud_pct"] = pd.to_numeric(df_acc_plot["exactitud"], errors="coerce") * 100
                fig_acc = px.bar(
                    df_acc_plot,
                    x="nombre",
                    y="exactitud_pct",
                    title="Exactitud (%) por Modelo Guardado",
                    labels={"nombre": "Modelo", "exactitud_pct": "Exactitud (%)"},
                )
                st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("Aún no hay modelos de árbol guardados en esta sesión.")

    with col_der:
        st.subheader("Historial de Reglas de Asociación")
        if historial_reglas:
            df_reglas_hist = pd.DataFrame(historial_reglas)
            st.dataframe(df_reglas_hist, use_container_width=True)
            if {"nombre", "reglas"}.issubset(df_reglas_hist.columns):
                fig_rules = px.bar(
                    df_reglas_hist,
                    x="nombre",
                    y="reglas",
                    title="Cantidad de Reglas por Ejecución Guardada",
                    labels={"nombre": "Ejecución", "reglas": "Reglas Generadas"},
                )
                st.plotly_chart(fig_rules, use_container_width=True)
        else:
            st.info("Aún no hay análisis de reglas guardados en esta sesión.")

    st.divider()

    # Vista previa y descarga de reporte consolidado y reglas vigentes.
    st.subheader("Exportación de Resultados")
    texto_reporte = generar_informe_texto(st.session_state)
    reglas_csv_df = obtener_reglas_exportables(st.session_state)

    st.text_area("Vista previa del reporte de sesión", value=texto_reporte, height=220)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="Descargar Reporte de Sesión (.txt)",
            data=texto_reporte.encode("utf-8"),
            file_name="reporte_sesion_mineria.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with d2:
        if not reglas_csv_df.empty:
            st.download_button(
                label="Descargar Reglas Actuales (.csv)",
                data=reglas_csv_df.to_csv(index=False).encode("utf-8"),
                file_name="reglas_asociacion_rf4.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button(
                "Descargar Reglas Actuales (.csv)",
                disabled=True,
                use_container_width=True,
                help="Genera primero reglas en el módulo de Reglas de Asociación.",
            )


if __name__ == "__main__":
    main()