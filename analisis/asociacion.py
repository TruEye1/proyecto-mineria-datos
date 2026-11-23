"""
Módulo: asociacion.py
Capa: Procesamiento (analisis/)
Descripción: Lógica independiente para la generación de itemsets frecuentes
y reglas de asociación con Apriori y FP-Growth (RF4 - mlxtend).
"""

import inspect
from typing import Dict, Any, List, Optional
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


def _formatear_intervalo(valor: Any) -> str:
    """Convierte intervalos de pandas en rangos legibles para las reglas."""
    if isinstance(valor, pd.Interval):
        izq = int(round(valor.left)) if abs(valor.left) >= 10 else round(valor.left, 2)
        der = int(round(valor.right)) if abs(valor.right) >= 10 else round(valor.right, 2)
        return f"{izq} a {der}"
    return str(valor)


def preparar_transacciones(
    df: pd.DataFrame,
    columnas_seleccionadas: List[str],
    incluir_numericas: bool = False,
    n_rangos: int = 3
) -> pd.DataFrame:
    """
    Transforma las columnas seleccionadas del DataFrame en una matriz booleana
    (formato transaccional One-Hot) lista para Apriori o FP-Growth.
    """
    if not columnas_seleccionadas:
        raise ValueError("Debes seleccionar al menos una columna para analizar.")

    cols_validas = [c for c in columnas_seleccionadas if c in df.columns]
    if not cols_validas:
        raise ValueError("Ninguna de las columnas seleccionadas existe en el dataset activo.")

    sub_df = df[cols_validas].copy()
    columnas_procesadas: Dict[str, pd.Series] = {}

    for col in cols_validas:
        if pd.api.types.is_numeric_dtype(sub_df[col]):
            if not incluir_numericas:
                continue

            serie_num = sub_df[col].dropna()
            if serie_num.nunique() > n_rangos:
                try:
                    cortes = pd.qcut(sub_df[col], q=n_rangos, duplicates="drop")
                except ValueError:
                    cortes = pd.cut(sub_df[col], bins=n_rangos)
                columnas_procesadas[col] = pd.Series(
                    [_formatear_intervalo(v) if pd.notna(v) else "Sin_Dato" for v in cortes],
                    index=sub_df.index
                )
            else:
                columnas_procesadas[col] = sub_df[col].fillna("Sin_Dato").astype(str)
        else:
            columnas_procesadas[col] = sub_df[col].fillna("Sin_Dato").astype(str).str.strip()

    if not columnas_procesadas:
        raise ValueError(
            "No hay columnas categóricas seleccionadas. Elige al menos una columna "
            "categórica o activa la opción 'Incluir variables numéricas discretizadas'."
        )

    df_proc = pd.DataFrame(columnas_procesadas, index=sub_df.index)
    df_transacciones = pd.get_dummies(df_proc, prefix_sep="=", dtype=bool)
    return df_transacciones


def generar_reglas_asociacion(
    df_transacciones: pd.DataFrame,
    algoritmo: str = "apriori",
    min_support: float = 0.25,
    metrica: str = "confidence",
    min_threshold: float = 0.6,
    max_len: Optional[int] = None
) -> Dict[str, Any]:
    """
    Ejecuta Apriori o FP-Growth sobre la matriz booleana y extrae reglas
    de asociación evaluadas por soporte, confianza y lift.
    """
    if df_transacciones.empty or df_transacciones.shape[1] == 0:
        raise ValueError("La matriz de transacciones está vacía.")

    alg = algoritmo.lower().strip()
    if alg == "fpgrowth":
        itemsets = fpgrowth(
            df_transacciones,
            min_support=min_support,
            use_colnames=True,
            max_len=max_len
        )
    else:
        itemsets = apriori(
            df_transacciones,
            min_support=min_support,
            use_colnames=True,
            max_len=max_len
        )

    if itemsets.empty:
        return {
            "itemsets": pd.DataFrame(columns=["items", "longitud", "support"]),
            "reglas": pd.DataFrame(),
            "total_transacciones": len(df_transacciones),
            "total_items_unicos": df_transacciones.shape[1],
            "total_itemsets": 0,
            "total_reglas": 0,
            "mejor_lift": 0.0
        }

    itemsets_ordenados = itemsets.sort_values(by="support", ascending=False).reset_index(drop=True)

    # Solo se pueden generar reglas si existe al menos un itemset con 2 o más elementos
    longitud_maxima = int(itemsets_ordenados["itemsets"].apply(len).max())
    reglas = pd.DataFrame()

    if longitud_maxima >= 2:
        try:
            params_ar = inspect.signature(association_rules).parameters
            kwargs_ar: Dict[str, Any] = {
                "metric": metrica,
                "min_threshold": min_threshold
            }
            if "num_itemsets" in params_ar:
                kwargs_ar["num_itemsets"] = len(itemsets_ordenados)

            reglas = association_rules(itemsets_ordenados, **kwargs_ar)
        except ValueError:
            reglas = pd.DataFrame()

    if not reglas.empty:
        reglas = reglas.copy()
        reglas["antecedentes"] = reglas["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
        reglas["consecuentes"] = reglas["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
        reglas["regla"] = reglas["antecedentes"] + "  ➔  " + reglas["consecuentes"]
        reglas = reglas.sort_values(
            by=["lift", "confidence", "support"],
            ascending=False
        ).reset_index(drop=True)
        mejor_lift = float(reglas["lift"].max())
    else:
        mejor_lift = 0.0

    itemsets_mostrar = itemsets_ordenados.copy()
    itemsets_mostrar["items"] = itemsets_mostrar["itemsets"].apply(lambda s: ", ".join(sorted(list(s))))
    itemsets_mostrar["longitud"] = itemsets_mostrar["itemsets"].apply(len)

    return {
        "itemsets": itemsets_mostrar[["items", "longitud", "support"]],
        "reglas": reglas,
        "total_transacciones": len(df_transacciones),
        "total_items_unicos": df_transacciones.shape[1],
        "total_itemsets": len(itemsets_mostrar),
        "total_reglas": len(reglas),
        "mejor_lift": mejor_lift
    }