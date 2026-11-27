"""
Módulo: generador_informe.py
Capa: Procesamiento (analisis/)
Descripción: Consolidación de resultados de la sesión y generación de reportes
exportables en texto plano y CSV.
"""

from datetime import datetime
from typing import Any, Dict, List
import pandas as pd


def _obtener_historial_arboles(session_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recupera la lista de modelos de árbol guardados en la sesión."""
    modelos = session_state.get("modelos_arbol")
    if isinstance(modelos, list):
        return modelos
    respaldo = session_state.get("historial_modelos")
    return respaldo if isinstance(respaldo, list) else []


def _obtener_historial_reglas(session_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recupera el historial de análisis de reglas de asociación guardados."""
    historial = session_state.get("historial_asociacion")
    return historial if isinstance(historial, list) else []


def construir_resumen_sesion(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae y resume el estado actual de carga, limpieza, árboles de decisión
    y reglas de asociación desde st.session_state.
    """
    df_original = session_state.get("df_original")
    df_limpio = session_state.get("df_limpio")
    historial_arboles = _obtener_historial_arboles(session_state)
    historial_reglas = _obtener_historial_reglas(session_state)
    reglas_actual = session_state.get("reglas_actual")

    resumen: Dict[str, Any] = {
        "generado_en": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tiene_dataset_original": df_original is not None,
        "tiene_dataset_limpio": df_limpio is not None,
        "filas_originales": int(df_original.shape[0]) if df_original is not None else 0,
        "columnas_originales": int(df_original.shape[1]) if df_original is not None else 0,
        "filas_limpias": int(df_limpio.shape[0]) if df_limpio is not None else 0,
        "columnas_limpias": int(df_limpio.shape[1]) if df_limpio is not None else 0,
        "total_modelos_arbol": len(historial_arboles),
        "total_analisis_reglas": len(historial_reglas),
        "mejor_exactitud_arbol": None,
        "mejor_lift_reglas": None,
    }

    if historial_arboles:
        exactitudes = [
            float(m["exactitud"])
            for m in historial_arboles
            if isinstance(m.get("exactitud"), (int, float))
        ]
        if exactitudes:
            resumen["mejor_exactitud_arbol"] = max(exactitudes)

    lifts: List[float] = []
    for item in historial_reglas:
        if isinstance(item.get("mejor_lift"), (int, float)):
            lifts.append(float(item["mejor_lift"]))
    if isinstance(reglas_actual, dict):
        res_actual = reglas_actual.get("resultado", {})
        if isinstance(res_actual.get("mejor_lift"), (int, float)):
            lifts.append(float(res_actual["mejor_lift"]))
    if lifts:
        resumen["mejor_lift_reglas"] = max(lifts)

    return resumen


def generar_informe_texto(session_state: Dict[str, Any]) -> str:
    """
    Genera un reporte consolidado en texto plano con los hitos registrados
    durante la sesión de trabajo.
    """
    resumen = construir_resumen_sesion(session_state)
    historial_arboles = _obtener_historial_arboles(session_state)
    historial_reglas = _obtener_historial_reglas(session_state)

    lineas: List[str] = [
        "REPORTE DE SESION - APLICACION DE MINERIA DE DATOS",
        f"Fecha y hora de generacion: {resumen['generado_en']}",
        "=" * 55,
        "",
        "1. ESTADO DE DATOS (CARGA Y LIMPIEZA)",
    ]

    if resumen["tiene_dataset_limpio"]:
        lineas.append(
            f"- Dataset limpio activo: {resumen['filas_limpias']} filas x "
            f"{resumen['columnas_limpias']} columnas."
        )
        if resumen["tiene_dataset_original"]:
            lineas.append(
                f"- Dataset original: {resumen['filas_originales']} filas x "
                f"{resumen['columnas_originales']} columnas."
            )
    else:
        lineas.append("- No hay un dataset limpio activo en la sesion.")

    lineas.extend(["", "2. MODELOS DE ARBOL DE DECISION (RF3)"])
    if historial_arboles:
        for modelo in historial_arboles:
            acc_val = modelo.get("exactitud")
            acc_txt = f"{float(acc_val) * 100:.2f}%" if isinstance(acc_val, (int, float)) else "N/A"
            lineas.append(
                f"- {modelo.get('nombre', 'Modelo')} | Objetivo: {modelo.get('target', '-')} | "
                f"Criterio: {modelo.get('criterio', '-')} | "
                f"Profundidad: {modelo.get('max_depth', '-')} | "
                f"Exactitud: {acc_txt}"
            )
    else:
        lineas.append("- Sin modelos guardados en el historial de esta sesion.")

    lineas.extend(["", "3. REGLAS DE ASOCIACION (RF4)"])
    if historial_reglas:
        for analisis in historial_reglas:
            lift_val = analisis.get("mejor_lift")
            lift_txt = f"{float(lift_val):.2f}" if isinstance(lift_val, (int, float)) else "0.00"
            lineas.append(
                f"- {analisis.get('nombre', 'Analisis')} | Algoritmo: {analisis.get('algoritmo', '-')} | "
                f"Soporte min: {analisis.get('min_support', '-')} | "
                f"Reglas: {analisis.get('reglas', 0)} | "
                f"Lift max: {lift_txt}"
            )
    else:
        lineas.append("- Sin analisis de reglas guardados en el historial de esta sesion.")

    return "\n".join(lineas)


def obtener_reglas_exportables(session_state: Dict[str, Any]) -> pd.DataFrame:
    """
    Retorna el DataFrame de reglas de asociación de la ejecución activa si existe,
    listo para exportar a CSV.
    """
    reglas_actual = session_state.get("reglas_actual")
    if not isinstance(reglas_actual, dict):
        return pd.DataFrame()

    resultado = reglas_actual.get("resultado", {})
    reglas_df = resultado.get("reglas")
    if not isinstance(reglas_df, pd.DataFrame) or reglas_df.empty:
        return pd.DataFrame()

    cols_preferidas = [
        c for c in ["antecedentes", "consecuentes", "support", "confidence", "lift"]
        if c in reglas_df.columns
    ]
    return reglas_df[cols_preferidas].copy()