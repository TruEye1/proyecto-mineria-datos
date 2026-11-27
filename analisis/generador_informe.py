"""
Módulo: generador_informe.py
Capa: Procesamiento (analisis/)
Descripción: Consolidación de resultados de la sesión y generación de reportes
exportables en texto plano, HTML autocontenido (RF5) y CSV.
"""

import html
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
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


def _obtener_reglas_activas(session_state: Dict[str, Any]) -> pd.DataFrame:
    """Obtiene el DataFrame de reglas activas desde reglas_actual o su clave de respaldo."""
    reglas_actual = session_state.get("reglas_actual")
    if isinstance(reglas_actual, dict):
        resultado = reglas_actual.get("resultado", {})
        if isinstance(resultado, dict):
            reglas_df = resultado.get("reglas")
            if isinstance(reglas_df, pd.DataFrame) and not reglas_df.empty:
                return reglas_df

    # Respaldo directo usado por la vista principal cuando no existe el contenedor completo.
    reglas_directas = session_state.get("reglas_asociacion")
    if isinstance(reglas_directas, pd.DataFrame) and not reglas_directas.empty:
        return reglas_directas

    return pd.DataFrame()


def obtener_reglas_exportables(session_state: Dict[str, Any]) -> pd.DataFrame:
    """
    Retorna el DataFrame de reglas de asociación de la ejecución activa si existe,
    listo para exportar a CSV.
    """
    reglas_df = _obtener_reglas_activas(session_state)
    if reglas_df.empty:
        return pd.DataFrame()

    cols_preferidas = [
        c for c in ["antecedentes", "consecuentes", "support", "confidence", "lift"]
        if c in reglas_df.columns
    ]
    return reglas_df[cols_preferidas].copy()


def generar_conclusiones_automaticas(session_state: Dict[str, Any]) -> List[str]:
    """Redacta hallazgos ejecutivos automáticos a partir del estado completo de la sesión."""
    conclusiones: List[str] = []

    df_limpio = session_state.get("df_limpio")
    resumen_limpieza = session_state.get("resumen_limpieza")
    modelos_arbol = _obtener_historial_arboles(session_state)
    historial_asociacion = _obtener_historial_reglas(session_state)
    reglas_activas = _obtener_reglas_activas(session_state)

    if isinstance(df_limpio, pd.DataFrame) and not df_limpio.empty:
        cols_num = list(df_limpio.select_dtypes(include=[np.number]).columns)
        n_cat = df_limpio.shape[1] - len(cols_num)
        conclusiones.append(
            f"El dataset limpio cuenta con {df_limpio.shape[0]} registros y {df_limpio.shape[1]} "
            f"variables ({len(cols_num)} numéricas y {n_cat} categóricas)."
        )
    else:
        conclusiones.append("Aún no se dispone de un dataset limpio para generar conclusiones de datos.")

    if isinstance(resumen_limpieza, dict) and resumen_limpieza:
        dup = int(resumen_limpieza.get("duplicados_eliminados", 0))
        nulos = sum((resumen_limpieza.get("nulos_imputados") or {}).values())
        atipicos = sum((resumen_limpieza.get("atipicos_tratados") or {}).values())
        conclusiones.append(
            f"Durante la limpieza automática se eliminaron {dup} duplicados, se imputaron {nulos} "
            f"valores nulos y se ajustaron {atipicos} valores atípicos mediante criterio IQR."
        )

    if modelos_arbol:
        mejor_modelo = max(
            modelos_arbol,
            key=lambda m: float(m.get("exactitud", 0.0)) if isinstance(m.get("exactitud"), (int, float)) else 0.0,
        )
        acc_pct = float(mejor_modelo.get("exactitud", 0.0)) * 100
        f1_pct = float(mejor_modelo.get("f1_score", 0.0)) * 100
        conclusiones.append(
            f"Se evaluaron {len(modelos_arbol)} modelo(s) de Árbol de Decisión; el mejor desempeño "
            f"corresponde a '{mejor_modelo.get('nombre', 'Modelo')}' con {acc_pct:.2f}% de exactitud "
            f"y {f1_pct:.2f}% de F1-Score."
        )
    else:
        conclusiones.append("No se han guardado modelos de Árbol de Decisión en esta sesión.")

    if not reglas_activas.empty and "lift" in reglas_activas.columns:
        top_regla = reglas_activas.sort_values(by="lift", ascending=False).iloc[0]
        conclusiones.append(
            f"Se generaron {len(reglas_activas)} reglas de asociación; la más fuerte es "
            f"'{top_regla.get('regla', '')}' (Soporte: {float(top_regla.get('support', 0.0)):.2%}, "
            f"Confianza: {float(top_regla.get('confidence', 0.0)):.2%}, Lift: {float(top_regla.get('lift', 0.0)):.2f})."
        )
    elif historial_asociacion:
        conclusiones.append(
            f"Existen {len(historial_asociacion)} análisis de asociación guardados en el historial de sesión."
        )
    else:
        conclusiones.append("No se han generado reglas de asociación en esta sesión.")

    return conclusiones


def _tabla_html_desde_registros(columnas: List[str], filas: List[List[str]]) -> str:
    """Construye una tabla HTML escapando encabezados y celdas para evitar inyección HTML."""
    ths = "".join(f"<th>{html.escape(str(col))}</th>" for col in columnas)
    trs: List[str] = []
    for fila in filas:
        tds = "".join(f"<td>{html.escape(str(val))}</td>" for val in fila)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _seccion_datos_html(session_state: Dict[str, Any]) -> str:
    """Genera los KPIs HTML del archivo cargado y dimensiones activas."""
    metadata = session_state.get("metadata_archivo")
    df_limpio = session_state.get("df_limpio")
    df_original = session_state.get("df_original")

    if not isinstance(metadata, dict) and df_limpio is None and df_original is None:
        return "<p class='muted'>Sin datos cargados en la sesión.</p>"

    meta = metadata if isinstance(metadata, dict) else {}
    df_activo = df_limpio if isinstance(df_limpio, pd.DataFrame) else df_original

    nombre = html.escape(str(meta.get("nombre_archivo", "Dataset en sesión")))
    encoding = html.escape(str(meta.get("encoding", "N/A")).upper())
    separador = html.escape(str(meta.get("nombre_separador", "N/A")))
    filas = int(df_activo.shape[0]) if isinstance(df_activo, pd.DataFrame) else int(meta.get("filas", 0))
    cols = int(df_activo.shape[1]) if isinstance(df_activo, pd.DataFrame) else int(meta.get("columnas", 0))

    return (
        "<div class='kpis'>"
        f"<div class='kpi'><span>Archivo</span><strong>{nombre}</strong></div>"
        f"<div class='kpi'><span>Registros activos</span><strong>{filas:,}</strong></div>"
        f"<div class='kpi'><span>Columnas</span><strong>{cols}</strong></div>"
        f"<div class='kpi'><span>Encoding / Sep</span><strong>{encoding} / {separador}</strong></div>"
        "</div>"
    )


def _seccion_limpieza_html(resumen_limpieza: Optional[Dict[str, Any]]) -> str:
    """Genera la tabla HTML con el detalle cuantitativo de la limpieza automática (RF2)."""
    if not isinstance(resumen_limpieza, dict) or not resumen_limpieza:
        return "<p class='muted'>No se registró ejecución de limpieza automática.</p>"

    nulos = resumen_limpieza.get("nulos_imputados") or {}
    atipicos = resumen_limpieza.get("atipicos_tratados") or {}
    tipos = resumen_limpieza.get("tipos_corregidos") or []

    filas_tabla = [
        [
            "Filas iniciales -> finales",
            f"{resumen_limpieza.get('filas_iniciales', 0)} -> {resumen_limpieza.get('filas_finales', 0)}",
        ],
        ["Duplicados eliminados", str(resumen_limpieza.get("duplicados_eliminados", 0))],
        ["Columnas normalizadas", "Sí" if resumen_limpieza.get("columnas_renombradas") else "No"],
        ["Tipos corregidos", ", ".join(str(t) for t in tipos) if tipos else "Ninguno"],
        ["Nulos imputados", ", ".join(f"{k}: {v}" for k, v in nulos.items()) if nulos else "Ninguno"],
        ["Atípicos tratados (IQR)", ", ".join(f"{k}: {v}" for k, v in atipicos.items()) if atipicos else "Ninguno"],
    ]
    return _tabla_html_desde_registros(["Proceso", "Detalle"], filas_tabla)


def _seccion_modelos_html(modelos_arbol: List[Dict[str, Any]]) -> str:
    """Genera la tabla HTML comparativa de modelos de Árbol de Decisión guardados (RF3)."""
    if not modelos_arbol:
        return "<p class='muted'>No hay modelos de Árbol de Decisión guardados en esta sesión.</p>"

    columnas = ["Nombre", "Objetivo", "Criterio", "Prof. Máx", "Exactitud", "Precisión", "Recall", "F1-Score"]
    filas: List[List[str]] = []
    for m in modelos_arbol:
        filas.append([
            str(m.get("nombre", "")),
            str(m.get("target", "")),
            str(m.get("criterio", "")),
            str(m.get("max_depth", "")),
            f"{float(m.get('exactitud', 0.0)) * 100:.2f}%",
            f"{float(m.get('precision', 0.0)) * 100:.2f}%",
            f"{float(m.get('recall', 0.0)) * 100:.2f}%",
            f"{float(m.get('f1_score', 0.0)) * 100:.2f}%",
        ])
    return _tabla_html_desde_registros(columnas, filas)


def _seccion_asociacion_html(
    historial_asociacion: List[Dict[str, Any]],
    reglas_activas: pd.DataFrame,
) -> str:
    """Genera las tablas HTML de reglas activas e historial de análisis (RF4)."""
    bloques: List[str] = []

    if not reglas_activas.empty:
        # Limita el informe a las 10 reglas con mayor lift para mantener legibilidad ejecutiva.
        top_reglas = (
            reglas_activas.sort_values(by="lift", ascending=False).head(10)
            if "lift" in reglas_activas.columns
            else reglas_activas.head(10)
        )
        filas_reglas = [
            [
                str(row.get("regla", "")),
                f"{float(row.get('support', 0.0)):.2%}",
                f"{float(row.get('confidence', 0.0)):.2%}",
                f"{float(row.get('lift', 0.0)):.2f}",
            ]
            for _, row in top_reglas.iterrows()
        ]
        bloques.append("<h3>Top 10 Reglas Activas (por Lift)</h3>")
        bloques.append(_tabla_html_desde_registros(["Regla", "Soporte", "Confianza", "Lift"], filas_reglas))

    if historial_asociacion:
        filas_hist = [
            [
                str(h.get("nombre", "")),
                str(h.get("algoritmo", "")),
                str(h.get("min_support", "")),
                str(h.get("itemsets", 0)),
                str(h.get("reglas", 0)),
                str(h.get("mejor_lift", 0.0)),
            ]
            for h in historial_asociacion
        ]
        bloques.append("<h3>Análisis Guardados en Sesión</h3>")
        bloques.append(_tabla_html_desde_registros(
            ["Nombre", "Algoritmo", "Soporte Mín.", "Itemsets", "Reglas", "Lift Máx."],
            filas_hist,
        ))

    if not bloques:
        return "<p class='muted'>No se generaron reglas de asociación en esta sesión.</p>"
    return "".join(bloques)


def generar_informe_html(session_state: Dict[str, Any]) -> str:
    """
    Construye un informe ejecutivo autocontenido en HTML5 con resumen de datos,
    limpieza automática, modelos de árbol de decisión, reglas de asociación y conclusiones.
    """
    fecha_gen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    resumen_limpieza = session_state.get("resumen_limpieza")
    modelos_arbol = _obtener_historial_arboles(session_state)
    historial_asociacion = _obtener_historial_reglas(session_state)
    reglas_activas = _obtener_reglas_activas(session_state)

    conclusiones = generar_conclusiones_automaticas(session_state)
    lista_conclusiones = "".join(f"<li>{html.escape(c)}</li>" for c in conclusiones)

    # Estilos embebidos para garantizar que el HTML descargado sea portable y autocontenido.
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Informe Ejecutivo de Minería de Datos</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 24px; line-height: 1.5; }}
  .container {{ max-width: 1020px; margin: 0 auto; background: #ffffff; border-radius: 10px; padding: 32px; box-shadow: 0 2px 12px rgba(15, 23, 42, 0.08); }}
  h1 {{ margin-top: 0; color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
  h2 {{ color: #0f172a; margin-top: 28px; border-left: 4px solid #2563eb; padding-left: 10px; }}
  h3 {{ color: #334155; margin-top: 18px; }}
  .meta {{ color: #64748b; font-size: 0.9rem; margin-bottom: 20px; }}
  .muted {{ color: #64748b; font-style: italic; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0; }}
  .kpi {{ background: #f1f5f9; border-radius: 8px; padding: 12px 14px; }}
  .kpi span {{ display: block; font-size: 0.8rem; color: #475569; }}
  .kpi strong {{ font-size: 1.05rem; color: #0f172a; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.92rem; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; }}
  th {{ background: #f1f5f9; color: #1e293b; font-weight: 600; }}
  tr:nth-child(even) td {{ background: #f8fafc; }}
  ul {{ padding-left: 20px; }}
  li {{ margin-bottom: 8px; }}
</style>
</head>
<body>
  <div class="container">
    <h1>Informe Ejecutivo de Minería de Datos</h1>
    <div class="meta">Generado el {html.escape(fecha_gen)}</div>

    <h2>1. Resumen de Datos</h2>
    {_seccion_datos_html(session_state)}

    <h2>2. Limpieza Automática (RF2)</h2>
    {_seccion_limpieza_html(resumen_limpieza)}

    <h2>3. Modelos de Árbol de Decisión (RF3)</h2>
    {_seccion_modelos_html(modelos_arbol)}

    <h2>4. Reglas de Asociación (RF4)</h2>
    {_seccion_asociacion_html(historial_asociacion, reglas_activas)}

    <h2>5. Conclusiones</h2>
    <ul>{lista_conclusiones}</ul>
  </div>
</body>
</html>"""