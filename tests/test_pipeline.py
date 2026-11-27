"""
Script de verificación técnica para el pipeline completo de RF1, RF2, RF3, RF4 y RF5.
Comprueba que los datos de prueba se cargan, se limpian (incluyendo atípicos),
que el árbol de decisión se entrena, evalúa y visualiza sin excepciones,
que las reglas de asociación (Apriori y FP-Growth) se generan correctamente,
y que los informes (texto plano, HTML autocontenido y CSV) cumplen RF5.
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
from analisis.asociacion import (
    preparar_transacciones,
    generar_reglas_asociacion
)
from analisis.generador_informe import (
    generar_conclusiones_automaticas,
    generar_informe_html,
    generar_informe_texto,
    obtener_reglas_exportables
)

def run_tests():
    print("=== INICIANDO TEST DEL PIPELINE ===")
    
    # 1. Test RF1: Carga y Validación
    csv_path = RUTA_RAIZ / "tests" / "dataset_prueba.csv"
    with open(csv_path, "rb") as f:
        resultado = validar_y_cargar_csv(f)
    
    assert resultado["valido"], f"RF1 falló: {resultado['errores']}"
    df_cargado = resultado["dataframe"]
    print(f"RF1 superado: Dataset cargado ({df_cargado.shape[0]} filas, {df_cargado.shape[1]} columnas)")

    # 2. Test RF2: Limpieza y Atípicos
    df_limpio, resumen = limpiar_datos(df_cargado)
    print(f"RF2 superado: Limpieza ejecutada.")
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
    print(f"RF3 Preparación superada: One-Hot features: {datos_prep['feature_names']}")

    modelo = entrenar_arbol(
        datos_prep["X_train"],
        datos_prep["y_train"],
        criterion="gini",
        max_depth=3,
        random_state=42
    )
    print(f"RF3 Entrenamiento superado: DecisionTreeClassifier instanciado y ajustado.")

    metricas = evaluar_modelo(
        modelo,
        datos_prep["X_test"],
        datos_prep["y_test"],
        datos_prep["class_names"],
        datos_prep["feature_names"]
    )
    print(f"RF3 Evaluación superada:")
    print(f"   - Exactitud (Accuracy): {metricas['exactitud'] * 100:.2f}%")
    print(f"   - Precisión: {metricas['precision'] * 100:.2f}%")
    print(f"   - Recall: {metricas['recall'] * 100:.2f}%")
    print(f"   - F1-Score: {metricas['f1_score'] * 100:.2f}%")
    print(f"   - Matriz de confusión:\n{metricas['matriz_confusion']}")

    reglas = exportar_arbol_a_reglas_texto(modelo, datos_prep["feature_names"])
    print(f"RF3 Reglas exportadas (primeras 3 líneas):\n" + "\n".join(reglas.splitlines()[:4]))

    # 4. Test RF4: Reglas de Asociación (Apriori y FP-Growth)
    cols_asoc = [
        c for c in df_limpio.columns
        if not pd.api.types.is_numeric_dtype(df_limpio[c])
    ]
    matriz_tx = preparar_transacciones(
        df_limpio,
        columnas_seleccionadas=cols_asoc,
        incluir_numericas=False
    )
    assert not matriz_tx.empty and matriz_tx.shape[1] > 0, "La matriz transaccional no debe estar vacía"

    res_apriori = generar_reglas_asociacion(
        matriz_tx,
        algoritmo="apriori",
        min_support=0.2,
        metrica="confidence",
        min_threshold=0.5
    )
    res_fpgrowth = generar_reglas_asociacion(
        matriz_tx,
        algoritmo="fpgrowth",
        min_support=0.2,
        metrica="confidence",
        min_threshold=0.5
    )

    assert res_apriori["total_itemsets"] > 0, "Apriori debe encontrar itemsets frecuentes"
    assert res_fpgrowth["total_itemsets"] > 0, "FP-Growth debe encontrar itemsets frecuentes"
    assert res_apriori["total_itemsets"] == res_fpgrowth["total_itemsets"], (
        "Apriori y FP-Growth deben encontrar la misma cantidad de itemsets"
    )
    assert res_apriori["total_reglas"] >= 0 and res_fpgrowth["total_reglas"] >= 0

    print(
        f"RF4 superado: Apriori ({res_apriori['total_itemsets']} itemsets, {res_apriori['total_reglas']} reglas) "
        f"y FP-Growth ({res_fpgrowth['total_itemsets']} itemsets, {res_fpgrowth['total_reglas']} reglas)."
    )

    # 5. Test RF5: Generación de informes (.txt, .html y .csv) y casos borde
    estado_sesion_prueba = {
        "df_original": df_cargado,
        "df_limpio": df_limpio,
        "metadata_archivo": resultado["metadata"],
        "resumen_limpieza": resumen,
        "modelos_arbol": [{
            "id": 1,
            "nombre": "Árbol Test <script>alert(1)</script>",
            "target": target,
            "criterio": "gini",
            "max_depth": 3,
            "exactitud": metricas["exactitud"],
            "precision": metricas["precision"],
            "recall": metricas["recall"],
            "f1_score": metricas["f1_score"],
        }],
        "historial_asociacion": [{
            "id": 1,
            "nombre": "Apriori Test",
            "algoritmo": "APRIORI",
            "min_support": 0.2,
            "itemsets": res_apriori["total_itemsets"],
            "reglas": res_apriori["total_reglas"],
            "mejor_lift": round(res_apriori["mejor_lift"], 2),
        }],
        "reglas_actual": {"resultado": res_apriori},
        "reglas_asociacion": res_apriori["reglas"],
    }

    conclusiones = generar_conclusiones_automaticas(estado_sesion_prueba)
    assert isinstance(conclusiones, list) and len(conclusiones) >= 4, (
        "RF5 debe generar al menos 4 conclusiones automáticas con sesión completa"
    )

    texto_informe = generar_informe_texto(estado_sesion_prueba)
    assert "REPORTE DE SESION" in texto_informe, "El reporte en texto debe incluir encabezado"

    reglas_csv = obtener_reglas_exportables(estado_sesion_prueba)
    assert isinstance(reglas_csv, pd.DataFrame), "Debe retornar un DataFrame exportable a CSV"

    html_informe = generar_informe_html(estado_sesion_prueba)
    assert html_informe.lstrip().startswith("<!DOCTYPE html>"), (
        "El informe HTML debe iniciar con <!DOCTYPE html>"
    )
    for seccion in [
        "Resumen de Datos",
        "Limpieza Automática",
        "Árbol de Decisión",
        "Reglas de Asociación",
        "Conclusiones",
    ]:
        assert seccion in html_informe, f"Falta sección esperada en informe HTML: {seccion}"

    assert "<script>alert(1)</script>" not in html_informe, (
        "El informe HTML debe escapar etiquetas peligrosas en nombres de modelos o datos"
    )
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_informe, (
        "El nombre escapado debe conservarse visible en el informe HTML"
    )

    # Caso borde: sesión completamente vacía no debe lanzar excepciones
    html_vacio = generar_informe_html({})
    conclusiones_vacias = generar_conclusiones_automaticas({})
    assert html_vacio.lstrip().startswith("<!DOCTYPE html>"), (
        "El informe HTML debe generarse de forma segura incluso con sesión vacía"
    )
    assert isinstance(conclusiones_vacias, list) and len(conclusiones_vacias) >= 1

    print(f"RF5 superado: Informe HTML ({len(html_informe)} bytes), texto, CSV y casos borde validados.")

    print("\n¡TODOS LOS TESTS UNITARIOS PASARON SATISFACTORIAMENTE!")

if __name__ == "__main__":
    run_tests()

