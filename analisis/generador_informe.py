"""
Módulo: generador_informe.py
Capa: Procesamiento (analisis/)
Descripción: Cálculo de estadísticas y correlaciones para el dashboard (RF6)
y generación de informes ejecutivos autocontenidos en HTML (RF5).
"""

import html
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def calcular_estadisticas_dashboard(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Calcula dimensiones, resumen numérico y matriz de correlación para el dashboard."""
    if df is None or df.empty:
        return {
            "filas": 0,
            "columnas": 0,
            "cols_numericas": [],
            "cols_categoricas": [],
            "resumen_numerico": pd.DataFrame(),
            "matriz_correlacion": pd.DataFrame(),
        }

    cols_numericas = list(df.select_dtypes(include=[np.number]).columns)
    cols_categoricas = [c for c in df.columns if c not in cols_numericas]

    resumen_numerico = (
        df[cols_numericas].describe().T.round(2) if cols_numericas else pd.DataFrame()
    )
    matriz_correlacion = (
        df[cols_numericas].corr().round(3) if len(cols_numericas) >= 2 else pd.DataFrame()
    )

    return {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "cols_numericas": cols_numericas,
        "cols_categoricas": cols_categoricas,
        "resumen_numerico": resumen_numerico,
        "matriz_correlacion": matriz_correlacion,
    }


def generar_conclusiones_automaticas(
    df: Optional[pd.DataFrame],
    resumen_limpieza: Optional[Dict[str, Any]],
    modelos_arbol: List[Dict[str, Any]],
    historial_asociacion: List[Dict[str, Any]],
    reglas_activas: Optional[pd.DataFrame],
) -> List[str]:
    """Redacta hallazgos clave a partir del dataset, limpieza, árboles y asociación."""
    conclusiones: List[str] = []

    if df is not None and not df.empty:
        stats = calcular_estadisticas_dashboard(df)
        conclusiones.append(
            f"El dataset procesado cuenta con {stats['filas']} registros y {stats['columnas']} "
            f"variables ({len(stats['cols_numericas'])} numéricas y {len(stats['cols_categoricas'])} categóricas)."
        )
    else:
        conclusiones.append("Aún no se dispone de un dataset limpio para generar conclusiones de datos.")

    if resumen_limpieza:
        dup = int(resumen_limpieza.get("duplicados_eliminados", 0))
        nulos = sum(resumen_limpieza.get("nulos_imputados", {}).values())
        atipicos = sum(resumen_limpieza.get("atipicos_tratados", {}).values())
        conclusiones.append(
            f"Durante la limpieza automática se eliminaron {dup} duplicados, se imputaron {nulos} "
            f"valores nulos y se ajustaron {atipicos} valores atípicos mediante criterio IQR."
        )

    if modelos_arbol:
        mejor_modelo = max(modelos_arbol, key=lambda m: float(m.get("exactitud", 0.0)))
        acc_pct = float(mejor_modelo.get("exactitud", 0.0)) * 100
        f1_pct = float(mejor_modelo.get("f1_score", 0.0)) * 100
        conclusiones.append(
            f"Se evaluaron {len(modelos_arbol)} modelo(s) de Árbol de Decisión; el mejor desempeño "
            f"corresponde a '{mejor_modelo.get('nombre', 'Modelo')}' con {acc_pct:.2f}% de exactitud "
            f"y {f1_pct:.2f}% de F1-Score."
        )
    else:
        conclusiones.append("No se han guardado modelos de Árbol de Decisión en esta sesión.")

    if reglas_activas is not None and not reglas_activas.empty:
        top_regla = reglas_activas.sort_values(by="lift", ascending=False).iloc[0]
        conclusiones.append(
            f"Se generaron {len(reglas_activas)} reglas de asociación; la más fuerte es "
            f"'{top_regla.get('regla', '')}' (Soporte: {float(top_regla.get('support', 0)):.2%}, "
            f"Confianza: {float(top_regla.get('confidence', 0)):.2%}, Lift: {float(top_regla.get('lift', 0)):.2f})."
        )
    elif historial_asociacion:
        conclusiones.append(
            f"Existen {len(historial_asociacion)} análisis de asociación guardados en el historial de sesión."
        )
    else:
        conclusiones.append("No se han generado reglas de asociación en esta sesión.")

    return conclusiones


def _tabla_html_desde_registros(columnas: List[str], filas: List[List[str]]) -> str:
    """Construye una tabla HTML con encabezados y celdas escapadas."""
    ths = "".join(f"<th>{html.escape(str(col))}</th>" for col in columnas)
    trs = []
    for fila in filas:
        tds = "".join(f"<td>{html.escape(str(val))}</td>" for val in fila)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _seccion_datos_html(
    metadata_archivo: Optional[Dict[str, Any]],
    df_limpio: Optional[pd.DataFrame],
) -> str:
    """Genera la sección HTML de resumen del dataset."""
    if not metadata_archivo and (df_limpio is None or df_limpio.empty):
        return "<p class='muted'>Sin datos cargados en la sesión.</p>"

    nombre = html.escape(str((metadata_archivo or {}).get("nombre_archivo", "Dataset en sesión")))
    encoding = html.escape(str((metadata_archivo or {}).get("encoding", "N/A")).upper())
    separador = html.escape(str((metadata_archivo or {}).get("nombre_separador", "N/A")))
    filas = int(df_limpio.shape[0]) if df_limpio is not None else int((metadata_archivo or {}).get("filas", 0))
    cols = int(df_limpio.shape[1]) if df_limpio is not None else int((metadata_archivo or {}).get("columnas", 0))

    return (
        f"<div class='kpis'>"
        f"<div class='kpi'><span>Archivo</span><strong>{nombre}</strong></div>"
        f"<div class='kpi'><span>Filas limpias</span><strong>{filas:,}</strong></div>"
        f"<div class='kpi'><span>Columnas</span><strong>{cols}</strong></div>"
        f"<div class='kpi'><span>Encoding / Sep</span><strong>{encoding} / {separador}</strong></div>"
        f"</div>"
    )


def _seccion_limpieza_html(resumen_limpieza: Optional[Dict[str, Any]]) -> str:
    """Genera la sección HTML con el detalle de la limpieza automática (RF2)."""
    if not resumen_limpieza:
        return "<p class='muted'>No se registró ejecución de limpieza automática.</p>"

    nulos = resumen_limpieza.get("nulos_imputados", {})
    atipicos = resumen_limpieza.get("atipicos_tratados", {})
    tipos = resumen_limpieza.get("tipos_corregidos", [])

    filas_tabla = [
        ["Filas iniciales -> finales", f"{resumen_limpieza.get('filas_iniciales', 0)} -> {resumen_limpieza.get('filas_finales', 0)}"],
        ["Duplicados eliminados", str(resumen_limpieza.get("duplicados_eliminados", 0))],
        ["Columnas normalizadas", "Sí" if resumen_limpieza.get("columnas_renombradas") else "No"],
        ["Tipos corregidos", ", ".join(tipos) if tipos else "Ninguno"],
        ["Nulos imputados", ", ".join(f"{k}: {v}" for k, v in nulos.items()) if nulos else "Ninguno"],
        ["Atípicos tratados (IQR)", ", ".join(f"{k}: {v}" for k, v in atipicos.items()) if atipicos else "Ninguno"],
    ]
    return _tabla_html_desde_registros(["Proceso", "Detalle"], filas_tabla)


def _seccion_modelos_html(modelos_arbol: List[Dict[str, Any]]) -> str:
    """Genera la sección HTML de modelos de Árbol de Decisión guardados (RF3)."""
    if not modelos_arbol:
        return "<p class='muted'>No hay modelos de Árbol de Decisión guardados en esta sesión.</p>"

    columnas = ["Nombre", "Objetivo", "Criterio", "Prof. Máx", "Exactitud", "Precisión", "Recall", "F1-Score"]
    filas = []
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
    reglas_activas: Optional[pd.DataFrame],
) -> str:
    """Genera la sección HTML de reglas de asociación e historial (RF4)."""
    bloques: List[str] = []

    if reglas_activas is not None and not reglas_activas.empty:
        top_reglas = reglas_activas.sort_values(by="lift", ascending=False).head(10)
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


def generar_html_informe(
    metadata_archivo: Optional[Dict[str, Any]],
    df_limpio: Optional[pd.DataFrame],
    resumen_limpieza: Optional[Dict[str, Any]],
    modelos_arbol: List[Dict[str, Any]],
    historial_asociacion: List[Dict[str, Any]],
    reglas_activas: Optional[pd.DataFrame],
) -> str:
    """Construye un informe ejecutivo autocontenido en formato HTML5."""
    fecha_gen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conclusiones = generar_conclusiones_automaticas(
        df=df_limpio,
        resumen_limpieza=resumen_limpieza,
        modelos_arbol=modelos_arbol,
        historial_asociacion=historial_asociacion,
        reglas_activas=reglas_activas,
    )
    lista_conclusiones = "".join(f"<li>{html.escape(c)}</li>" for c in conclusiones)

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
    {_seccion_datos_html(metadata_archivo, df_limpio)}

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
