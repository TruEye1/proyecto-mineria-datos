"""
Módulo: arbol_decision.py
Capa: Procesamiento (analisis/)
Descripción: Lógica completa e independiente para el entrenamiento, evaluación y 
visualización de modelos de Árboles de Decisión (RF3 - Scikit-learn).
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt

def preparar_datos(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """Codifica variables y separa los datos en conjuntos de entrenamiento y prueba."""
    # Comprueba que la variable objetivo y al menos una predictora sean válidas.
    if target_col not in df.columns:
        raise ValueError(f"La variable objetivo '{target_col}' no existe en el DataFrame.")
    
    if not feature_cols:
        raise ValueError("Debe seleccionar al menos una variable predictora.")
    
    features_validas = [c for c in feature_cols if c in df.columns and c != target_col]
    if not features_validas:
        raise ValueError("No se encontraron variables predictoras válidas distintas a la objetivo.")

    # Separa predictores y objetivo antes de codificarlos para el modelo.
    X_crudo = df[features_validas].copy()
    y_crudo = df[target_col].copy()

    # Convierte categorías a indicadores y etiquetas de clase a enteros.
    X_encoded = pd.get_dummies(X_crudo, drop_first=False, dtype=int)
    feature_names = list(X_encoded.columns)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_crudo.astype(str))
    class_names = [str(c) for c in label_encoder.classes_]

    # Estratifica solo si cada clase tiene suficientes ejemplos para aparecer en ambos subconjuntos.
    conteo_clases = pd.Series(y_encoded).value_counts()
    usar_estratificacion = bool(conteo_clases.min() >= 2)

    stratify_param = y_encoded if usar_estratificacion else None

    # Divide de forma reproducible los predictores y objetivos codificados.
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "class_names": class_names,
        "label_encoder": label_encoder,
        "target_col": target_col,
        "features_seleccionadas": features_validas,
        "estratificado": usar_estratificacion
    }

def entrenar_arbol(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    criterion: str = "gini",
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> DecisionTreeClassifier:
    """Configura y ajusta un clasificador de árbol con los datos de entrenamiento."""
    # Sustituye criterios no admitidos por el valor predeterminado del proyecto.
    criterios_validos = ["gini", "entropy", "log_loss"]
    if criterion not in criterios_validos:
        criterion = "gini"

    # Construye el estimador con los hiperparámetros seleccionados y lo entrena.
    modelo = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )

    modelo.fit(X_train, y_train)
    return modelo

def evaluar_modelo(
    modelo: DecisionTreeClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: List[str],
    feature_names: List[str]
) -> Dict[str, Any]:
    """Calcula métricas, matriz de confusión, reporte e importancia de variables."""
    y_pred = modelo.predict(X_test)

    # Limita el reporte a las clases presentes en las etiquetas reales o predichas.
    clases_presentes = np.unique(np.concatenate((y_test, y_pred)))
    nombres_presentes = [class_names[int(c)] for c in clases_presentes]

    # Calcula métricas ponderadas para resumir el rendimiento global del clasificador.
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Genera diagnósticos por clase y la matriz que compara valores reales y predichos.
    matriz = confusion_matrix(y_test, y_pred)
    
    reporte = classification_report(
        y_test,
        y_pred,
        labels=clases_presentes,
        target_names=nombres_presentes,
        zero_division=0,
        output_dict=False
    )

    # Asocia la importancia interna del árbol a los nombres de las variables.
    importancias = modelo.feature_importances_
    df_importancia = pd.DataFrame({
        "variable": feature_names,
        "importancia": importancias
    }).sort_values(by="importancia", ascending=False).reset_index(drop=True)

    # Entrega todos los resultados necesarios para las vistas de la aplicación.
    return {
        "exactitud": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "matriz_confusion": matriz,
        "reporte_clasificacion": reporte,
        "importancia_variables": df_importancia,
        "total_test": len(y_test),
        "y_pred": y_pred
    }

def generar_figura_arbol(
    modelo: DecisionTreeClassifier,
    feature_names: List[str],
    class_names: List[str],
    max_depth_viz: Optional[int] = 3,
    figsize: Tuple[int, int] = (16, 9)
) -> plt.Figure:
    """Dibuja el árbol y limita los niveles visibles para facilitar su lectura."""
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    
    # Ajusta los nombres de las clases al orden interno usado por el estimador.
    nombres_modelo = [class_names[int(c)] for c in modelo.classes_]
    
    plot_tree(
        modelo,
        feature_names=feature_names,
        class_names=nombres_modelo,
        max_depth=max_depth_viz,
        filled=True,
        rounded=True,
        proportion=True,
        precision=2,
        fontsize=9,
        ax=ax
    )
    ax.set_title("Estructura del Árbol de Decisión", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig

def exportar_arbol_a_reglas_texto(
    modelo: DecisionTreeClassifier,
    feature_names: List[str]
) -> str:
    """Exporta las decisiones aprendidas como reglas de texto legibles."""
    return export_text(modelo, feature_names=feature_names)