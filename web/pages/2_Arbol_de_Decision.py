"""
Módulo: 2_Arbol_de_Decision.py
Capa: Presentación (web/pages/)
Descripción: Interfaz interactiva para el entrenamiento, ajuste de hiperparámetros,
evaluación y gestión de modelos de Árboles de Decisión (RF3).

Cumple estrictamente con los requerimientos del enunciado:
- Selección de variable objetivo y predictoras.
- Configuración de hiperparámetros (profundidad, criterio, división train/test).
- Visualización gráfica del árbol y reglas textuales.
- Métricas: Exactitud, Matriz de Confusión, Precisión, Recall y F1-Score.
- Gestión de modelos: Guardar en session_state, listar, comparar y eliminar.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff

# Asegurar que la raíz del proyecto esté en el sys.path para importar desde analisis/
RUTA_RAIZ = Path(__file__).resolve().parent.parent.parent
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from analisis.arbol_decision import (
    preparar_datos,
    entrenar_arbol,
    evaluar_modelo,
    generar_figura_arbol,
    exportar_arbol_a_reglas_texto
)


def inicializar_estado():
    if "df_limpio" not in st.session_state:
        st.session_state["df_limpio"] = None
    if "modelos_arbol" not in st.session_state:
        st.session_state["modelos_arbol"] = []
    if "arbol_actual" not in st.session_state:
        st.session_state["arbol_actual"] = None


def render_matriz_confusion(matriz: np.ndarray, class_names: list):
    """Renderiza la matriz de confusión interactiva con Plotly."""
    z = matriz.tolist()
    x = [f"Pred: {c}" for c in class_names]
    y = [f"Real: {c}" for c in class_names]

    # Invertir eje Y para que coincida con la orientación matemática habitual
    fig = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        colorscale="Blues",
        showscale=True
    )
    fig.update_layout(
        title_text="<b>Matriz de Confusión (Valores Absolutos)</b>",
        xaxis_title="Clase Predicha por el Modelo",
        yaxis_title="Clase Real de la Muestra",
        margin=dict(t=50, l=80, r=50, b=50),
        height=400
    )
    return fig


def render_importancia_variables(df_importancia: pd.DataFrame):
    """Renderiza el gráfico de barras horizontales con la importancia de variables."""
    df_filtrado = df_importancia[df_importancia["importancia"] > 0].copy()
    if df_filtrado.empty:
        df_filtrado = df_importancia.head(5).copy()

    fig = px.bar(
        df_filtrado.sort_values(by="importancia", ascending=True),
        x="importancia",
        y="variable",
        orientation="h",
        labels={"importancia": "Reducción Media de Impureza (MDI)", "variable": "Predictora"},
        title="<b>Importancia Relativa de Variables Predictoras</b>",
        color="importancia",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(height=380, margin=dict(t=40, l=100, r=40, b=40))
    return fig


def main():
    st.set_page_config(
        page_title="Árbol de Decisión | Minería de Datos",
        page_icon="🌳",
        layout="wide"
    )
    inicializar_estado()

    st.title("🌳 Módulo 3: Árbol de Decisión (RF3)")
    st.markdown(
        """
        Este módulo implementa algoritmos supervisados de **Árboles de Decisión** (Scikit-learn), 
        permitiendo entrenar clasificadores interpretables, calibrar hiperparámetros y comparar modelos guardados.
        """
    )

    # 1. Validación de prerrequisito (Carga y Limpieza)
    if st.session_state["df_limpio"] is None:
        st.warning("⚠️ **Prerrequisito no cumplido:** Debes cargar y limpiar un dataset primero.")
        st.info("👉 Ve al módulo **1. Carga y Limpieza** en el menú lateral para subir tu archivo.")
        if st.session_state.get("df_original") is not None:
            st.info("💡 Ya subiste un archivo, pero aún no presionas el botón **Ejecutar Limpieza Automática**.")
        return

    df = st.session_state["df_limpio"]

    st.success(f"✅ **Dataset activo:** {df.shape[0]} filas × {df.shape[1]} columnas disponibles para modelado.")

    st.divider()

    # 2. Configuración de Variables (Target y Features)
    st.subheader("1. Selección de Variables")
    col_target, col_features = st.columns([1, 2])

    columnas_disponibles = list(df.columns)

    with col_target:
        # Sugerir la última columna como objetivo por defecto habitual en data mining
        indice_target_default = len(columnas_disponibles) - 1
        target_col = st.selectbox(
            "🎯 Variable Objetivo (Target / Clase):",
            options=columnas_disponibles,
            index=indice_target_default,
            help="Columna que el modelo intentará predecir (ej. aprobado, categoría, compra)."
        )

    with col_features:
        posibles_features = [c for c in columnas_disponibles if c != target_col]
        
        seleccionar_todas = st.checkbox("Seleccionar todas las predictoras disponibles", value=True)
        default_features = posibles_features if seleccionar_todas else posibles_features[:3]

        feature_cols = st.multiselect(
            "📊 Variables Predictoras (Features / Atributos):",
            options=posibles_features,
            default=default_features,
            help="Variables de entrada que usará el árbol para tomar decisiones."
        )

    if not feature_cols:
        st.error("Debes seleccionar al menos una variable predictora para entrenar.")
        return

    st.divider()

    # 3. Configuración de Hiperparámetros
    st.subheader("2. Calibración de Hiperparámetros")
    st.markdown("Ajusta los parámetros para controlar la complejidad y evitar el sobreajuste (*overfitting*):")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        criterio = st.selectbox(
            "Criterio de División:",
            options=["gini", "entropy", "log_loss"],
            format_func=lambda x: "Gini (Impureza)" if x == "gini" else ("Entropía (Ganancia Info)" if x == "entropy" else "Log Loss"),
            help="Métrica estadística para evaluar la calidad de cada división en los nodos."
        )

    with c2:
        usar_limite_prof = st.checkbox("Limitar profundidad máxima", value=True)
        if usar_limite_prof:
            max_depth = st.slider("Profundidad Máxima (max_depth):", min_value=1, max_value=20, value=3, step=1)
        else:
            max_depth = None
            st.caption("Sin límite (árbol crecerá hasta hojas puras).")

    with c3:
        min_samples_split = st.slider(
            "Mínimo muestras para dividir:",
            min_value=2,
            max_value=20,
            value=2,
            step=1,
            help="Número mínimo de muestras que debe tener un nodo interno para ser dividido."
        )

    with c4:
        test_size_pct = st.slider(
            "Porcentaje de Test:",
            min_value=10,
            max_value=50,
            value=25,
            step=5,
            help="Proporción del dataset reservada para validar el rendimiento."
        )

    st.divider()

    # 4. Botón de Entrenamiento
    btn_entrenar = st.button("🚀 Entrenar Árbol de Decisión", type="primary", use_container_width=True)

    if btn_entrenar:
        with st.spinner("Preparando datos y ajustando árbol de decisión..."):
            try:
                # Capa de Procesamiento
                datos_preparados = preparar_datos(
                    df=df,
                    target_col=target_col,
                    feature_cols=feature_cols,
                    test_size=test_size_pct / 100.0,
                    random_state=42
                )

                modelo = entrenar_arbol(
                    X_train=datos_preparados["X_train"],
                    y_train=datos_preparados["y_train"],
                    criterion=criterio,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )

                metricas = evaluar_modelo(
                    modelo=modelo,
                    X_test=datos_preparados["X_test"],
                    y_test=datos_preparados["y_test"],
                    class_names=datos_preparados["class_names"],
                    feature_names=datos_preparados["feature_names"]
                )

                # Guardar resultado del entrenamiento activo en el session_state
                st.session_state["arbol_actual"] = {
                    "modelo": modelo,
                    "datos_preparados": datos_preparados,
                    "metricas": metricas,
                    "hiperparametros": {
                        "criterio": criterio,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "test_size": test_size_pct / 100.0
                    },
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                st.success("🎉 ¡Modelo entrenado y evaluado con éxito!")

            except Exception as e:
                st.error(f"❌ Error durante el entrenamiento: {str(e)}")

    # 5. Visualización de Resultados del Modelo Activo
    if st.session_state["arbol_actual"] is not None:
        resultado = st.session_state["arbol_actual"]
        m = resultado["metricas"]
        hp = resultado["hiperparametros"]
        dp = resultado["datos_preparados"]

        st.subheader("3. Métricas de Rendimiento en Conjunto de Prueba")
        st.caption(f"Evaluado sobre **{m['total_test']} muestras de prueba** (Split estratificado: {'Sí' if dp['estratificado'] else 'No'}).")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Exactitud (Accuracy)", f"{m['exactitud'] * 100:.2f}%", help="Porcentaje total de clasificaciones correctas.")
        col_m2.metric("Precisión (Ponderada)", f"{m['precision'] * 100:.2f}%", help="Mide qué tan confiables son las predicciones positivas.")
        col_m3.metric("Recall / Sensibilidad", f"{m['recall'] * 100:.2f}%", help="Mide la capacidad de encontrar todos los casos positivos reales.")
        col_m4.metric("F1-Score", f"{m['f1_score'] * 100:.2f}%", help="Media armónica entre precisión y recall.")

        st.divider()

        # Pestañas de análisis detallado
        tab_arbol, tab_matriz, tab_importancia, tab_guardar = st.tabs([
            "🌳 Visualización del Árbol",
            "📊 Matriz de Confusión y Diagnóstico",
            "📈 Importancia de Variables",
            "💾 Gestión y Guardado del Modelo"
        ])

        with tab_arbol:
            st.markdown("#### Estructura Gráfica del Árbol")
            col_viz_ctrl, _ = st.columns([1, 2])
            with col_viz_ctrl:
                max_depth_viz = st.slider(
                    "Profundidad visual a renderizar:",
                    min_value=1,
                    max_value=max(1, hp["max_depth"] or 5),
                    value=min(3, max(1, hp["max_depth"] or 3)),
                    help="Permite limitar los niveles visibles para evitar saturación visual."
                )

            fig_arbol = generar_figura_arbol(
                modelo=resultado["modelo"],
                feature_names=dp["feature_names"],
                class_names=dp["class_names"],
                max_depth_viz=max_depth_viz
            )
            st.pyplot(fig_arbol, clear_figure=True)

            with st.expander("📄 Ver Reglas de Decisión en Formato Texto (IF-THEN)"):
                reglas_texto = exportar_arbol_a_reglas_texto(resultado["modelo"], dp["feature_names"])
                st.code(reglas_texto, language="text")

        with tab_matriz:
            col_mat, col_rep = st.columns([1, 1])
            with col_mat:
                fig_matriz = render_matriz_confusion(m["matriz_confusion"], dp["class_names"])
                st.plotly_chart(fig_matriz, use_container_width=True)
            with col_rep:
                st.markdown("#### Reporte de Clasificación Detallado")
                st.code(m["reporte_clasificacion"], language="text")

        with tab_importancia:
            st.markdown("#### Relevancia de Variables Predictoras (MDI)")
            st.markdown(
                "La métrica **Mean Decrease in Impurity (MDI)** cuantifica cuánto contribuye cada variable "
                "a reducir la incertidumbre estadística total en los nodos del árbol."
            )
            fig_imp = render_importancia_variables(m["importancia_variables"])
            st.plotly_chart(fig_imp, use_container_width=True)
            st.dataframe(m["importancia_variables"], use_container_width=True)

        with tab_guardar:
            st.markdown("#### Guardar Modelo en la Sesión")
            st.markdown(
                "Guarda este modelo entrenado para compararlo contra otras configuraciones o utilizarlo en el informe ejecutivo."
            )
            
            c_nom, c_btn = st.columns([2, 1])
            with c_nom:
                nombre_defecto = f"Árbol ({hp['criterio'].title()}) - Prof {hp['max_depth'] or 'Inf'} [{resultado['timestamp']}]"
                nombre_modelo = st.text_input("Nombre identificador del modelo:", value=nombre_defecto)
            
            with c_btn:
                st.write("")
                st.write("")
                if st.button("💾 Guardar en Memoria de Sesión", type="primary"):
                    nuevo_registro = {
                        "id": len(st.session_state["modelos_arbol"]) + 1,
                        "nombre": nombre_modelo,
                        "target": dp["target_col"],
                        "criterio": hp["criterio"],
                        "max_depth": hp["max_depth"] or "Ilimitada",
                        "test_size": hp["test_size"],
                        "exactitud": m["exactitud"],
                        "precision": m["precision"],
                        "recall": m["recall"],
                        "f1_score": m["f1_score"],
                        "total_variables": len(dp["feature_names"]),
                        "timestamp": resultado["timestamp"]
                    }
                    st.session_state["modelos_arbol"].append(nuevo_registro)
                    st.success(f"✅ ¡Modelo '{nombre_modelo}' guardado correctamente!")

    # 6. Tabla Comparativa de Modelos Guardados (Requisito PDF: "guardar, listar, comparar y eliminar")
    st.divider()
    st.subheader("📋 Modelos Guardados en la Sesión")

    if st.session_state["modelos_arbol"]:
        df_modelos = pd.DataFrame(st.session_state["modelos_arbol"])
        
        # Formatear porcentajes para visualización
        df_mostrar = df_modelos.copy()
        for col_met in ["exactitud", "precision", "recall", "f1_score"]:
            df_mostrar[col_met] = df_mostrar[col_met].apply(lambda x: f"{x * 100:.2f}%")

        st.dataframe(df_mostrar, use_container_width=True)

        col_del1, col_del2 = st.columns([1, 4])
        with col_del1:
            if st.button("🗑️ Borrar Historial de Modelos"):
                st.session_state["modelos_arbol"] = []
                st.rerun()
    else:
        st.info("Aún no has guardado ningún modelo en esta sesión de trabajo.")


if __name__ == "__main__":
    main()

