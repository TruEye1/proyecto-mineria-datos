import csv
import io
from typing import Dict, Any, Tuple, Optional
import pandas as pd

TAMANO_MAXIMO_MB: float = 50.0
ENCODINGS_SOPORTADOS: list = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
SEPARADORES_CANDIDATOS: list = [",", ";", "\t", "|"]

def validar_extension(nombre_archivo: str) -> Tuple[bool, str]:
    if not nombre_archivo:
        return False, "No se proporcionó un nombre de archivo válido."
    if not nombre_archivo.lower().endswith(".csv"):
        return False, f"Formato no válido ('{nombre_archivo}'). Solo se admiten archivos .csv."
    return True, "Extensión .csv válida."

def validar_tamano(tamano_bytes: int, max_mb: float = TAMANO_MAXIMO_MB) -> Tuple[bool, float, str]:
    if tamano_bytes <= 0:
        return False, 0.0, "El archivo está completamente vacío (0 bytes)."
    
    tamano_mb = round(tamano_bytes / (1024 * 1024), 4)
    if tamano_mb > max_mb:
        return False, tamano_mb, f"El archivo ({tamano_mb:.2f} MB) excede el máximo de {max_mb} MB."
    
    return True, tamano_mb, f"Tamaño válido ({tamano_mb:.4f} MB)."

def detectar_encoding(contenido_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    muestra_bytes = contenido_bytes[:65536]
    for encoding in ENCODINGS_SOPORTADOS:
        try:
            texto_decodificado = muestra_bytes.decode(encoding)
            return encoding, texto_decodificado
        except UnicodeDecodeError:
            continue
    return None, None

def detectar_separador(muestra_texto: str) -> str:
    lineas = [linea for linea in muestra_texto.splitlines() if linea.strip()]
    if not lineas:
        return ","

    muestra_lineas = "\n".join(lineas[:20])
    try:
        dialecto = csv.Sniffer().sniff(muestra_lineas, delimiters="".join(SEPARADORES_CANDIDATOS))
        return dialecto.delimiter
    except csv.Error:
        encabezado = lineas[0]
        conteos = {sep: encabezado.count(sep) for sep in SEPARADORES_CANDIDATOS}
        mejor_separador = max(conteos, key=conteos.get)
        return mejor_separador if conteos[mejor_separador] > 0 else ","

def validar_y_cargar_csv(
    archivo_subido: Any,
    max_mb: float = TAMANO_MAXIMO_MB
) -> Dict[str, Any]:
    
    resultado: Dict[str, Any] = {
        "valido": False,
        "mensaje": "",
        "errores": [],
        "dataframe": None,
        "metadata": {
            "nombre_archivo": getattr(archivo_subido, "name", "desconocido.csv"),
            "tamano_mb": 0.0,
            "encoding": None,
            "separador": None,
            "nombre_separador": None,
            "filas": 0,
            "columnas": 0,
            "nombres_columnas": []
        }
    }

    nombre_archivo = resultado["metadata"]["nombre_archivo"]
    ext_ok, msg_ext = validar_extension(nombre_archivo)
    if not ext_ok:
        resultado["errores"].append(msg_ext)
        resultado["mensaje"] = msg_ext
        return resultado

    try:
        contenido_bytes = archivo_subido.getvalue()
    except AttributeError:
        archivo_subido.seek(0)
        contenido_bytes = archivo_subido.read()

    tam_ok, tamano_mb, msg_tam = validar_tamano(len(contenido_bytes), max_mb)
    resultado["metadata"]["tamano_mb"] = tamano_mb
    if not tam_ok:
        resultado["errores"].append(msg_tam)
        resultado["mensaje"] = msg_tam
        return resultado

    encoding, muestra_texto = detectar_encoding(contenido_bytes)
    if not encoding or not muestra_texto:
        msg_enc = "No fue posible reconocer la codificación del archivo."
        resultado["errores"].append(msg_enc)
        resultado["mensaje"] = msg_enc
        return resultado
    resultado["metadata"]["encoding"] = encoding

    separador = detectar_separador(muestra_texto)
    nombres_sep = {",": "Coma (,)", ";": "Punto y coma (;)", "\t": "Tabulación (\\t)", "|": "Tubería (|)"}
    resultado["metadata"]["separador"] = separador
    resultado["metadata"]["nombre_separador"] = nombres_sep.get(separador, f"Otro ({separador})")

    try:
        buffer_bytes = io.BytesIO(contenido_bytes)
        df = pd.read_csv(buffer_bytes, encoding=encoding, sep=separador)

        if df.empty or len(df.columns) == 0:
            msg_vacio = "El archivo CSV no contiene registros o columnas válidas."
            resultado["errores"].append(msg_vacio)
            resultado["mensaje"] = msg_vacio
            return resultado

        if len(df.columns) < 2:
            msg_cols = "Se detectó solo 1 columna. Verifique el archivo y el separador."
            resultado["errores"].append(msg_cols)
            resultado["mensaje"] = msg_cols
            return resultado

        resultado["valido"] = True
        resultado["dataframe"] = df
        resultado["metadata"]["filas"] = int(df.shape[0])
        resultado["metadata"]["columnas"] = int(df.shape[1])
        resultado["metadata"]["nombres_columnas"] = list(df.columns)
        resultado["mensaje"] = "Archivo validado y cargado exitosamente."

    except Exception as e:
        msg_err = f"Error inesperado al leer el archivo CSV: {str(e)}"
        resultado["errores"].append(msg_err)
        resultado["mensaje"] = msg_err

    return resultado