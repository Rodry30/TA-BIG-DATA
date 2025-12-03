import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve, auc
)


def encontrar_source_path(base_dir):
    candidates = [
        os.path.join(base_dir, 'data_warehouse', 'dataset_imputado.csv'),
        os.path.join(base_dir, 'dataset_imputado.csv')
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def encontrar_modelo_path():
    candidates = [
        os.path.join('modelos', 'rf_model.joblib'),
        os.path.join('data', 'modelos', 'rf_model.joblib'),
        '/opt/spark-data/modelos/rf_model.joblib'
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No se encontró el modelo en: {candidates}")


def main():
    # Rutas
    if os.path.exists('/opt/spark-data'):
        base = '/opt/spark-data'
        report_dir = '/opt/spark-data/data_warehouse/graficos'
    else:
        base = 'data'
        report_dir = os.path.join('data', 'data_warehouse', 'graficos')

    os.makedirs(report_dir, exist_ok=True)

    # Leer dataset
    source = encontrar_source_path(base)
    print(f"Leyendo dataset: {source}")
    df = pd.read_csv(source)

    target = 'condicion_ocupacion'
    if target not in df.columns:
        raise SystemExit(f"Columna '{target}' no encontrada")

    df = df[df[target].notnull()]

    # Definir features (igual a entrenamiento)
    numeric_cols = ['edad']
    categorical_cols = [
        'sexo', 'nivel_educativo_cod', 'lengua', 'Etnia',
        'Concentracion', 'Socializar', 'Experiencia', 'Movilidad'
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    X = df[numeric_cols + categorical_cols]
    y = df[target]

    # Cargar modelo
    model_path = encontrar_modelo_path()
    print(f"Cargando modelo: {model_path}")
    model = joblib.load(model_path)

    # Predecir
    print("Realizando predicciones...")
    y_pred = model.predict(X)
    y_proba = None
    try:
        y_proba = model.predict_proba(X)
    except Exception:
        pass

    # Calcular metricas
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

    # Matriz de confusión
    cm = confusion_matrix(y, y_pred)
    classes = np.unique(y)

    # FP y FN
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)

    # Reporte de clasificación
    report = classification_report(y, y_pred, zero_division=0)

    # Crear reporte de texto
    report_text = f"""
=== ANÁLISIS DE EFICACIA DEL MODELO RANDOM FOREST ===
Fecha: 2025-12-03

MÉTRICAS GLOBALES
{'='*50}
Accuracy (Exactitud):        {acc:.4f}
Precision (Ponderado):       {prec:.4f}
Recall (Sensibilidad Pond.): {rec:.4f}
F1-Score (Ponderado):        {f1:.4f}

MATRIZ DE CONFUSIÓN
{'='*50}
Clases: {', '.join(map(str, classes))}

{cm}

ANÁLISIS DETALLADO POR CLASE
{'='*50}
"""

    for i, cls in enumerate(classes):
        report_text += f"\nClase: {cls}\n"
        report_text += f"  Verdaderos Positivos (TP):  {tp[i]}\n"
        report_text += f"  Falsos Positivos (FP):      {fp[i]}\n"
        report_text += f"  Falsos Negativos (FN):      {fn[i]}\n"
        report_text += f"  Verdaderos Negativos (TN):  {tn[i]}\n"

    report_text += f"\n\nREPORTE DE CLASIFICACIÓN DETALLADO\n{'='*50}\n{report}\n"

    # Distribución de clases en datos originales
    class_dist = y.value_counts(normalize=True) * 100
    report_text += f"\nDISTRIBUCIÓN DE CLASES EN DATOS\n{'='*50}\n"
    for cls, pct in class_dist.items():
        report_text += f"{cls}: {pct:.2f}%\n"

    # Distribución de predicciones
    pred_dist = pd.Series(y_pred).value_counts(normalize=True) * 100
    report_text += f"\nDISTRIBUCIÓN DE PREDICCIONES\n{'='*50}\n"
    for cls, pct in pred_dist.items():
        report_text += f"{cls}: {pct:.2f}%\n"

    # Guardar reporte
    report_file = os.path.join(report_dir, 'analisis_eficacia_modelo.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"✅ Reporte guardado: {report_file}")

    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Matriz de confusión (heatmap)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=classes, yticklabels=classes, cbar=True)
    axes[0, 0].set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Verdadero')
    axes[0, 0].set_xlabel('Predicción')

    # 2. Distribución de clases reales vs predichas
    x_pos = np.arange(len(classes))
    width = 0.35
    real_counts = [sum(y == cls) for cls in classes]
    pred_counts = [sum(y_pred == cls) for cls in classes]
    axes[0, 1].bar(x_pos - width/2, real_counts, width, label='Real', alpha=0.8)
    axes[0, 1].bar(x_pos + width/2, pred_counts, width, label='Predicción', alpha=0.8)
    axes[0, 1].set_xlabel('Clase')
    axes[0, 1].set_ylabel('Cantidad')
    axes[0, 1].set_title('Distribución de Clases: Real vs Predicción', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(classes)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. Métricas por clase
    from sklearn.metrics import precision_recall_fscore_support
    prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(y, y_pred, zero_division=0)
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    x_pos_metrics = np.arange(len(classes))
    width_metrics = 0.25
    axes[1, 0].bar(x_pos_metrics - width_metrics, prec_per_class, width_metrics, label='Precision', alpha=0.8)
    axes[1, 0].bar(x_pos_metrics, rec_per_class, width_metrics, label='Recall', alpha=0.8)
    axes[1, 0].bar(x_pos_metrics + width_metrics, f1_per_class, width_metrics, label='F1-Score', alpha=0.8)
    axes[1, 0].set_xlabel('Clase')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Métricas por Clase', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x_pos_metrics)
    axes[1, 0].set_xticklabels(classes)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. Falsos positivos y falsos negativos
    fp_fn_data = pd.DataFrame({
        'FP (Falsos Positivos)': fp,
        'FN (Falsos Negativos)': fn
    }, index=classes)
    fp_fn_data.plot(kind='bar', ax=axes[1, 1], alpha=0.8)
    axes[1, 1].set_title('Falsos Positivos vs Falsos Negativos', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Cantidad')
    axes[1, 1].set_xlabel('Clase')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plot_file = os.path.join(report_dir, 'analisis_eficacia_graficos.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Gráficos guardados: {plot_file}")
    plt.close()

    # Resumen en consola
    print("\n" + "="*60)
    print("RESUMEN DE ANÁLISIS")
    print("="*60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nTotal de muestras: {len(y)}")
    for cls in classes:
        count = sum(y == cls)
        pct = (count / len(y)) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
