import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def limpiar_datos(df_original: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df_original.copy()
    
    resumen = {
        "filas_iniciales": len(df),
        "columnas_iniciales": len(df.columns),
        "duplicados_eliminados": 0,
        "nulos_imputados": {},
        "columnas_renombradas": False,
        "tipos_corregidos": []
    }

    columnas_originales = list(df.columns)
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    if list(df.columns) != columnas_originales:
        resumen["columnas_renombradas"] = True

    filas_antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    resumen["duplicados_eliminados"] = filas_antes - len(df)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
            try:
                temp_col = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(temp_col)
                resumen["tipos_corregidos"].append(col)
            except ValueError:
                pass 

    for col in df.columns:
        nulos_col = int(df[col].isnull().sum())
        if nulos_col > 0:
            resumen["nulos_imputados"][col] = nulos_col
            
            if pd.api.types.is_numeric_dtype(df[col]):
                mediana = df[col].median()
                df[col] = df[col].fillna(mediana)
            else:
                moda = df[col].mode()
                valor_imputacion = moda[0] if not moda.empty else "Desconocido"
                df[col] = df[col].fillna(valor_imputacion)

    resumen["filas_finales"] = len(df)
    resumen["columnas_finales"] = len(df.columns)

    return df, resumen