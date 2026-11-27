"""
Módulo: 2_Arbol_de_Decision.py
Capa: Presentación (web/pages/)
Descripción: Interfaz interactiva para el entrenamiento, ajuste de hiperparámetros,
evaluación y gestión de modelos de Árboles de Decisión (RF3).
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

RUTA_RAIZ = Path(__file__).resolve().parent.parent.parent
if str(RUTA_RAIZ) not in sys.path:
    # Permite importar la lógica de análisis al ejecutar esta página directamente.
    sys.path.insert(0, str(RUTA_RAIZ))

from analisis.arbol_decision import (
    preparar_datos,
    entrenar_arbol,
    evaluar_modelo,
    generar_figura_arbol,
    exportar_arbol_a_reglas_texto
)


def inicializar_estado():
    """Crea en la sesión las claves que usa la página sin sobrescribir datos existentes."""
    if "df_limpio" not in st.session_state:
        st.session_state["df_limpio"] = None
    if "modelos_arbol" not in st.session_state:
        st.session_state["modelos_arbol"] = []
    if "arbol_actual" not in st.session_state:
        st.session_state["arbol_actual"] = None


def eliminar_modelo_por_id(modelos: List[Dict[str, Any]], id_modelo: int) -> List[Dict[str, Any]]:
    """Retorna la lista de modelos excluyendo el registro con el ID indicado."""
    return [modelo for modelo in modelos if modelo.get("id") != id_modelo]


def render_matriz_confusion(matriz: np.ndarray, class_names: list):
    """Renderiza la matriz de confusión interactiva de forma segura y robusta."""
    
    # Ajusta las etiquetas a la cantidad de clases representadas en la matriz.
    n_clases_reales = matriz.shape[0]
    nombres_ejes = class_names[:n_clases_reales] if len(class_names) >= n_clases_reales else [f"C{i}" for i in range(n_clases_reales)]
    
    # Construye el gráfico con las clases reales y predichas en sus respectivos ejes.
    x_labels = [f"Pred: {c}" for c in nombres_ejes]
    y_labels = [f"Real: {c}" for c in nombres_ejes]

    fig = px.imshow(
        matriz,
        labels=dict(x="Clase Predicha", y="Clase Real", color="Frecuencia"),
        x=x_labels,
        y=y_labels,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto"
    )
    fig.update_layout(
        title_text="<b>Matriz de Confusión</b>",
        margin=dict(t=50, l=80, r=50, b=50),
        height=400
    )
    return fig


def render_importancia_variables(df_importancia: pd.DataFrame):
    """Crea un gráfico horizontal de la importancia de cada variable predictora."""
    # Oculta variables sin aporte; si todas son cero, muestra algunas para dar contexto.
    df_filtrado = df_importancia[df_importancia["importancia"] > 0].copy()
    if df_filtrado.empty:
        df_filtrado = df_importancia.head(5).copy()

    # Ordena las variables para que las de mayor importancia queden al final del eje.
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


def render_comparacion_modelos(df_modelos: pd.DataFrame):
    """Genera un gráfico de barras agrupadas comparando métricas entre modelos guardados."""
    etiquetas_metricas = {
        "exactitud": "Exactitud",
        "precision": "Precisión",
        "recall": "Recall",
        "f1_score": "F1-Score"
    }
    df_melt = df_modelos.melt(
        id_vars=["nombre"],
        value_vars=list(etiquetas_metricas.keys()),
        var_name="metrica",
        value_name="valor"
    )
    df_melt["metrica"] = df_melt["metrica"].map(etiquetas_metricas)
    df_melt["porcentaje"] = (df_melt["valor"] * 100).round(2)

    fig = px.bar(
        df_melt,
        x="nombre",
        y="porcentaje",
        color="metrica",
        barmode="group",
        text_auto=".1f",
        labels={
            "nombre": "Modelo Guardado",
            "porcentaje": "Rendimiento (%)",
            "metrica": "Métrica"
        },
        title="<b>Comparativa de Rendimiento entre Modelos (%)</b>"
    )
    fig.update_layout(
        height=400,
        yaxis_range=[0, 105],
        margin=dict(t=50, l=50, r=40, b=60),
        legend_title_text="Métrica"
    )
    return fig


def main():
    # Configura la página y prepara las claves que conservarán los datos entre interacciones.
    st.set_page_config(
        page_title="Árbol de Decisión | Minería de Datos",
        page_icon="🌳",
        layout="wide"
    )
    inicializar_estado()

    st.title("Módulo 3: Árbol de Decisión (RF3)")
    st.markdown(
        """
        Este módulo implementa algoritmos supervisados de **Árboles de Decisión** (Scikit-learn), 
        permitiendo entrenar clasificadores interpretables, calibrar hiperparámetros y comparar modelos guardados.
        """
    )

    # Impide entrenar hasta que exista un conjunto de datos ya limpiado.
    if st.session_state["df_limpio"] is None:
        st.warning("**Prerrequisito no cumplido:** Debes cargar y limpiar un dataset primero.")
        st.info("Ve al módulo **1. Carga y Limpieza** en el menú lateral para subir tu archivo.")
        if st.session_state.get("df_original") is not None:
            st.info("Ya subiste un archivo, pero aún no presionas el botón **Ejecutar Limpieza Automática**.")
        return

    df = st.session_state["df_limpio"]

    st.success(f"**Dataset activo:** {df.shape[0]} filas × {df.shape[1]} columnas disponibles para modelado.")

    st.divider()

    # Permite elegir la variable a predecir y las variables que usará el modelo.
    st.subheader("1. Selección de Variables")
    col_target, col_features = st.columns([1, 2])

    columnas_disponibles = list(df.columns)

    with col_target:
        indice_target_default = len(columnas_disponibles) - 1
        target_col = st.selectbox(
            "Variable Objetivo (Target / Clase):",
            options=columnas_disponibles,
            index=indice_target_default,
            help="Columna que el modelo intentará predecir (ej. aprobado, categoría, compra)."
        )

    with col_features:
        posibles_features = [c for c in columnas_disponibles if c != target_col]
        
        seleccionar_todas = st.checkbox("Seleccionar todas las predictoras disponibles", value=True)
        default_features = posibles_features if seleccionar_todas else posibles_features[:3]

        feature_cols = st.multiselect(
            "Variables Predictoras (Features / Atributos):",
            options=posibles_features,
            default=default_features,
            help="Variables de entrada que usará el árbol para tomar decisiones."
        )

    if not feature_cols:
        st.error("Debes seleccionar al menos una variable predictora para entrenar.")
        return

    st.divider()

    # Recoge las opciones que controlan la complejidad del árbol y la evaluación.
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

    # Prepara los datos, entrena el árbol y guarda el resultado para mostrarlo después.
    btn_entrenar = st.button("Entrenar Árbol de Decisión", type="primary", use_container_width=True)

    if btn_entrenar:
        # Evita mostrar resultados anteriores mientras se genera el nuevo modelo.
        st.session_state["arbol_actual"] = None
        
        with st.spinner("Preparando datos y ajustando árbol de decisión..."):
            try:
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
                st.success("¡Modelo entrenado y evaluado con éxito!")

            except Exception as e:
                st.error(f"Error durante el entrenamiento: {str(e)}")

    # Muestra métricas y visualizaciones solo cuando hay un entrenamiento disponible.
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

        # Separa los resultados en vistas para explorar el árbol, diagnóstico y gestión.
        tab_arbol, tab_matriz, tab_importancia, tab_guardar = st.tabs([
            "Visualización del Árbol",
            "Matriz de Confusión y Diagnóstico",
            "Importancia de Variables",
            "Gestión y Guardado del Modelo"
        ])

        with tab_arbol:
            # Muestra el árbol con profundidad visual ajustable y sus reglas IF-THEN.
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

            with st.expander("Ver Reglas de Decisión en Formato Texto (IF-THEN)"):
                reglas_texto = exportar_arbol_a_reglas_texto(resultado["modelo"], dp["feature_names"])
                st.code(reglas_texto, language="text")

        with tab_matriz:
            # Contrasta predicciones y clases reales junto al reporte por clase.
            col_mat, col_rep = st.columns([1, 1])
            with col_mat:
                fig_matriz = render_matriz_confusion(m["matriz_confusion"], dp["class_names"])
                st.plotly_chart(fig_matriz, use_container_width=True)
            with col_rep:
                st.markdown("#### Reporte de Clasificación Detallado")
                st.code(m["reporte_clasificacion"], language="text")

        with tab_importancia:
            # Presenta el aporte relativo de las variables calculado durante la evaluación.
            st.markdown("#### Relevancia de Variables Predictoras (MDI)")
            st.markdown(
                "La métrica **Mean Decrease in Impurity (MDI)** cuantifica cuánto contribuye cada variable "
                "a reducir la incertidumbre estadística total en los nodos del árbol."
            )
            fig_imp = render_importancia_variables(m["importancia_variables"])
            st.plotly_chart(fig_imp, use_container_width=True)
            st.dataframe(m["importancia_variables"], use_container_width=True)

        with tab_guardar:
            # Conserva un resumen del modelo entrenado en el historial de esta sesión.
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
                if st.button("Guardar en Memoria de Sesión", type="primary"):
                    nuevo_id = max(
                        (mod.get("id", 0) for mod in st.session_state["modelos_arbol"]),
                        default=0
                    ) + 1
                    nuevo_registro = {
                        "id": nuevo_id,
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
                    st.success(f"¡Modelo '{nombre_modelo}' guardado correctamente!")

    st.divider()
    # Lista los modelos guardados y permite borrar el historial de la sesión.
    st.subheader("Modelos Guardados en la Sesión")

    if st.session_state["modelos_arbol"]:
        df_modelos = pd.DataFrame(st.session_state["modelos_arbol"])
        
        df_mostrar = df_modelos.copy()
        for col_met in ["exactitud", "precision", "recall", "f1_score"]:
            df_mostrar[col_met] = df_mostrar[col_met].apply(lambda x: f"{x * 100:.2f}%")

        st.dataframe(df_mostrar, use_container_width=True)

        if len(df_modelos) >= 2:
            st.markdown("#### Comparación Visual de Modelos")
            nombres_disponibles = df_modelos["nombre"].tolist()
            modelos_comp = st.multiselect(
                "Modelos a comparar:",
                options=nombres_disponibles,
                default=nombres_disponibles
            )
            if len(modelos_comp) >= 2:
                df_comp = df_modelos[df_modelos["nombre"].isin(modelos_comp)]
                fig_comp = render_comparacion_modelos(df_comp)
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("Selecciona al menos 2 modelos para visualizar la comparación.")

        opciones_modelos = {
            mod["id"]: f"#{mod['id']} — {mod['nombre']}"
            for mod in st.session_state["modelos_arbol"]
        }
        id_modelo_sel = st.selectbox(
            "Seleccionar modelo guardado:",
            options=list(opciones_modelos.keys()),
            format_func=lambda idx: opciones_modelos[idx]
        )

        col_del_ind, col_del_all, _ = st.columns([1, 1, 2])
        with col_del_ind:
            if st.button("Eliminar modelo seleccionado", use_container_width=True):
                st.session_state["modelos_arbol"] = eliminar_modelo_por_id(
                    st.session_state["modelos_arbol"],
                    id_modelo_sel
                )
                st.rerun()
        with col_del_all:
            if st.button("Borrar Historial de Modelos"):
                st.session_state["modelos_arbol"] = []
                st.rerun()
    else:
        st.info("Aún no has guardado ningún modelo en esta sesión de trabajo.")


if __name__ == "__main__":
    main()