"""
Módulo: arbol_decision.py
Capa: Procesamiento (analisis/)
Descripción: Lógica completa e independiente para el entrenamiento, evaluación y 
visualización de modelos de Árboles de Decisión (RF3 - Scikit-learn).

Principios pedagógicos para la defensa técnica:
1. Desacoplamiento estricto: Esta capa no depende de Streamlit ni de la interfaz visual.
2. Preparación matemática: Codificación One-Hot para variables categóricas (evita relaciones
   ordinales artificiales) y división estratificada (preserva la proporción de clases).
3. Criterios de división: Soporta Gini (impureza estadística) y Entropía (ganancia de información).
4. Métricas globales: Exactitud (Accuracy), Precisión ponderada, Recall ponderado, F1-Score,
   Matriz de Confusión y reducción media de impureza (Feature Importance MDI).
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
    """
    Preprocesa y divide los datos para entrenamiento supervisado.
    
    Pasos:
    1. Valida que las columnas existan y que no haya intersección entre target y features.
    2. Codifica variables predictoras categóricas usando One-Hot Encoding (pd.get_dummies).
    3. Codifica la variable objetivo si es categórica mediante LabelEncoder.
    4. Aplica train_test_split de forma estratificada para mantener la distribución de clases.
    
    Retorna un diccionario con matrices de train/test, nombres de columnas y encoders.
    """
    if target_col not in df.columns:
        raise ValueError(f"La variable objetivo '{target_col}' no existe en el DataFrame.")
    
    if not feature_cols:
        raise ValueError("Debe seleccionar al menos una variable predictora.")
    
    features_validas = [c for c in feature_cols if c in df.columns and c != target_col]
    if not features_validas:
        raise ValueError("No se encontraron variables predictoras válidas distintas a la objetivo.")

    # 1. Separar X e y
    X_crudo = df[features_validas].copy()
    y_crudo = df[target_col].copy()

    # 2. Codificación One-Hot para variables predictoras no numéricas
    # drop_first=False para mantener total interpretabilidad en las ramas del árbol
    X_encoded = pd.get_dummies(X_crudo, drop_first=False, dtype=int)
    feature_names = list(X_encoded.columns)

    # 3. Codificación de la variable objetivo (target)
    label_encoder = None
    if not pd.api.types.is_numeric_dtype(y_crudo) or y_crudo.dtype == "object":
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_crudo.astype(str))
        class_names = [str(c) for c in label_encoder.classes_]
    else:
        # Si es numérica discreta (ej. 0 y 1), ordenamos sus valores únicos
        valores_unicos = np.unique(y_crudo.dropna())
        y_encoded = y_crudo.to_numpy()
        class_names = [str(val) for val in valores_unicos]

    # Validar mínimo de muestras por clase para split estratificado
    conteo_clases = pd.Series(y_encoded).value_counts()
    usar_estratificacion = bool(conteo_clases.min() >= 2)

    stratify_param = y_encoded if usar_estratificacion else None

    # 4. División Train / Test
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
    """
    Instancia y ajusta un modelo DecisionTreeClassifier de Scikit-learn.
    
    Parámetros ajustables:
    - criterion: 'gini' o 'entropy' (criterio de ganancia de información)
    - max_depth: Profundidad máxima del árbol (previene sobreajuste / overfitting)
    - min_samples_split: Cantidad mínima de registros para dividir un nodo interno
    - min_samples_leaf: Cantidad mínima de registros en una hoja final
    """
    criterios_validos = ["gini", "entropy", "log_loss"]
    if criterion not in criterios_validos:
        criterion = "gini"

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
    """
    Calcula métricas rigurosas de rendimiento para el modelo sobre el conjunto de test.
    
    Retorna:
    - exactitud (Accuracy): proporción global de aciertos
    - precision (Precision ponderada): minimización de falsos positivos
    - recall (Sensibilidad ponderada): tasa de detección de verdaderos positivos
    - f1_score: balance armónico entre precisión y recall
    - matriz_confusion: matriz bidimensional numpy (reales vs predichos)
    - reporte_texto: tabla de métricas por clase
    - importancia_variables: diccionario ordenado con la importancia de cada feature
    - total_muestras_test: cantidad de instancias evaluadas
    """
    y_pred = modelo.predict(X_test)

    # Métricas globales
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    matriz = confusion_matrix(y_test, y_pred)
    reporte = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=False
    )

    # Importancia de variables (MDI: Mean Decrease in Impurity)
    importancias = modelo.feature_importances_
    df_importancia = pd.DataFrame({
        "variable": feature_names,
        "importancia": importancias
    }).sort_values(by="importancia", ascending=False).reset_index(drop=True)

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
    """
    Genera una figura de Matplotlib renderizando la estructura gráfica del árbol de decisión.
    
    Parámetros:
    - max_depth_viz: Permite acotar la profundidad visual para que sea legible en Streamlit
    - filled=True: Colorea los nodos según la clase mayoritaria (azul, naranja, verde, etc.)
    - rounded=True: Bordes curvos profesionales estilo Orange Data Mining.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    
    plot_tree(
        modelo,
        feature_names=feature_names,
        class_names=class_names,
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
    """
    Exporta la lógica del árbol a formato de reglas textuales (IF-THEN).
    Ideal para auditoría humana y para incluir en el reporte HTML (RF5).
    """
    return export_text(modelo, feature_names=feature_names)

