"""
Script de verificación técnica para el pipeline completo de RF1, RF2 y RF3.
Comprueba que los datos de prueba se cargan, se limpian (incluyendo atípicos),
y que el árbol de decisión se entrena, evalúa y visualiza sin excepciones.
"""

import sys
from pathlib import Path
import pandas as pd

RUTA_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUTA_RAIZ))

from analisis.validador import validar_y_cargar_csv
from analisis.limpiador import limpiar_datos
from analisis.arbol_decision import (
    preparar_datos,
    entrenar_arbol,
    evaluar_modelo,
    exportar_arbol_a_reglas_texto
)

def run_tests():
    print("=== INICIANDO TEST DEL PIPELINE ===")
    
    # 1. Test RF1: Carga y Validación
    csv_path = RUTA_RAIZ / "tests" / "dataset_prueba.csv"
    with open(csv_path, "rb") as f:
        resultado = validar_y_cargar_csv(f)
    
    assert resultado["valido"], f"RF1 falló: {resultado['errores']}"
    df_cargado = resultado["dataframe"]
    print(f"✅ RF1 superado: Dataset cargado ({df_cargado.shape[0]} filas, {df_cargado.shape[1]} columnas)")

    # 2. Test RF2: Limpieza y Atípicos
    df_limpio, resumen = limpiar_datos(df_cargado)
    print(f"✅ RF2 superado: Limpieza ejecutada.")
    print(f"   - Filas iniciales: {resumen['filas_iniciales']} -> Finales: {resumen['filas_finales']}")
    print(f"   - Duplicados eliminados: {resumen['duplicados_eliminados']}")
    print(f"   - Nulos imputados en: {list(resumen['nulos_imputados'].keys())}")
    print(f"   - Atípicos tratados (IQR): {resumen['atipicos_tratados']}")
    assert resumen["duplicados_eliminados"] > 0, "Debe detectar duplicados en el dataset de prueba"
    assert len(resumen["atipicos_tratados"]) > 0, "Debe detectar atípicos en el dataset de prueba"

    # 3. Test RF3: Preparación de datos, entrenamiento y evaluación
    target = "aprobado"
    features = [c for c in df_limpio.columns if c != target and c != "id_cliente"]
    
    datos_prep = preparar_datos(df_limpio, target_col=target, feature_cols=features, test_size=0.25)
    print(f"✅ RF3 Preparación superada: One-Hot features: {datos_prep['feature_names']}")

    modelo = entrenar_arbol(
        datos_prep["X_train"],
        datos_prep["y_train"],
        criterion="gini",
        max_depth=3,
        random_state=42
    )
    print(f"✅ RF3 Entrenamiento superado: DecisionTreeClassifier instanciado y ajustado.")

    metricas = evaluar_modelo(
        modelo,
        datos_prep["X_test"],
        datos_prep["y_test"],
        datos_prep["class_names"],
        datos_prep["feature_names"]
    )
    print(f"✅ RF3 Evaluación superada:")
    print(f"   - Exactitud (Accuracy): {metricas['exactitud'] * 100:.2f}%")
    print(f"   - Precisión: {metricas['precision'] * 100:.2f}%")
    print(f"   - Recall: {metricas['recall'] * 100:.2f}%")
    print(f"   - F1-Score: {metricas['f1_score'] * 100:.2f}%")
    print(f"   - Matriz de confusión:\n{metricas['matriz_confusion']}")

    reglas = exportar_arbol_a_reglas_texto(modelo, datos_prep["feature_names"])
    print(f"✅ RF3 Reglas exportadas (primeras 3 líneas):\n" + "\n".join(reglas.splitlines()[:4]))

    print("\n🎉 ¡TODOS LOS TESTS UNITARIOS PASARON SATISFACTORIAMENTE!")

if __name__ == "__main__":
    run_tests()
