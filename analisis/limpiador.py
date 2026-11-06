"""
Módulo: limpiador.py
Capa: Procesamiento (analisis/)
Descripción: Lógica de limpieza y preprocesamiento automático de datos (RF2).
Cumple con los 6 requerimientos mínimos de limpieza:
1. Normalización de nombres de columnas.
2. Eliminación de filas duplicadas.
3. Corrección automática de tipos de datos mal formateados.
4. Tratamiento inteligente de nulos (imputación por mediana en numéricos y moda en categóricos).
5. Detección y tratamiento de valores atípicos (outliers) mediante el método IQR (Rango Intercuartílico).
6. Resumen estructurado de transformaciones realizadas.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def detectar_y_tratar_atipicos_iqr(
    df: pd.DataFrame, factor_iqr: float = 1.5
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Detecta y trata valores atípicos (outliers) en columnas numéricas continuas
    utilizando el método del Rango Intercuartílico (IQR / Criterio de Tukey).
    
    Explicación pedagógica para la defensa:
    - Se calculan el primer cuartil Q1 (percentil 25) y tercer cuartil Q3 (percentil 75).
    - El IQR representa la dispersión del 50% central de los datos (Q3 - Q1).
    - Límites: [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    - Tratamiento: Winsorización (capping), es decir, recortar los valores al límite
      permitido. Esto neutraliza la distorsión de valores aberrantes sin destruir
      filas completas que contienen información valiosa en otras columnas.
    """
    df_procesado = df.copy()
    conteo_atipicos: Dict[str, int] = {}

    columnas_numericas = df_procesado.select_dtypes(include=[np.number]).columns

    for col in columnas_numericas:
        # Si la columna tiene muy pocos valores únicos (ej. flags binarias 0/1), no aplicar IQR
        if df_procesado[col].nunique() <= 2:
            continue

        q1 = df_procesado[col].quantile(0.25)
        q3 = df_procesado[col].quantile(0.75)
        iqr = q3 - q1

        # Si el IQR es mayor a 0 (hay varianza en los datos)
        if iqr > 0:
            limite_inferior = q1 - (factor_iqr * iqr)
            limite_superior = q3 + (factor_iqr * iqr)

            mascara_atipicos = (df_procesado[col] < limite_inferior) | (df_procesado[col] > limite_superior)
            total_atipicos = int(mascara_atipicos.sum())

            if total_atipicos > 0:
                conteo_atipicos[col] = total_atipicos
                # Aplicamos capping / winsorización con clip
                df_procesado[col] = df_procesado[col].clip(lower=limite_inferior, upper=limite_superior)

    return df_procesado, conteo_atipicos


def limpiar_datos(df_original: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ejecuta el pipeline completo de limpieza automática sobre una copia del DataFrame.
    
    Retorna:
        - pd.DataFrame: Dataset limpio listo para modelado (RF3/RF4).
        - dict: Resumen cuantitativo de cada modificación efectuada.
    """
    df = df_original.copy()

    resumen: Dict[str, Any] = {
        "filas_iniciales": len(df),
        "columnas_iniciales": len(df.columns),
        "duplicados_eliminados": 0,
        "nulos_imputados": {},
        "columnas_renombradas": False,
        "tipos_corregidos": [],
        "atipicos_tratados": {},
        "filas_finales": 0,
        "columnas_finales": 0
    }

    # 1. Normalización de nombres de columnas
    columnas_originales = list(df.columns)
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    if list(df.columns) != columnas_originales:
        resumen["columnas_renombradas"] = True

    # 2. Eliminación de registros duplicados exactos
    filas_antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    resumen["duplicados_eliminados"] = filas_antes - len(df)

    # 3. Corrección automática de tipos de datos mal formateados
    # Detecta cadenas de texto que en realidad son números con coma decimal o espacios
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
            try:
                temp_col = df[col].astype(str).str.replace(",", ".")
                df[col] = pd.to_numeric(temp_col)
                resumen["tipos_corregidos"].append(col)
            except (ValueError, TypeError):
                pass

    # 4. Tratamiento de valores nulos mediante imputación estadística
    for col in df.columns:
        nulos_col = int(df[col].isnull().sum())
        if nulos_col > 0:
            resumen["nulos_imputados"][col] = nulos_col

            if pd.api.types.is_numeric_dtype(df[col]):
                # En variables numéricas se usa la mediana por ser robusta ante asimetrías
                mediana = df[col].median()
                df[col] = df[col].fillna(mediana)
            else:
                # En categóricas se usa la moda (valor más frecuente)
                moda = df[col].mode()
                valor_imputacion = moda[0] if not moda.empty else "Desconocido"
                df[col] = df[col].fillna(valor_imputacion)

    # 5. Detección y manejo de valores atípicos (outliers) - RF2
    df, atipicos_dict = detectar_y_tratar_atipicos_iqr(df)
    resumen["atipicos_tratados"] = atipicos_dict

    # 6. Métricas finales
    resumen["filas_finales"] = len(df)
    resumen["columnas_finales"] = len(df.columns)

    return df, resumen
