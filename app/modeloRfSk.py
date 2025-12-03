import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def encontrar_source_path(base_dir):
    candidates = [
        os.path.join(base_dir, 'data_warehouse', 'dataset_imputado.csv'),
        os.path.join(base_dir, 'dataset_imputado.csv')
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def main():
    # configurar rutas
    if os.path.exists('/opt/spark-data'):
        base = '/opt/spark-data'
        out_dir = '/opt/spark-data/data_warehouse/graficos'
        model_dir = '/opt/spark-data/modelos'
    else:
        base = 'data'
        out_dir = os.path.join('data', 'data_warehouse', 'graficos')
        model_dir = 'modelos'

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    source = encontrar_source_path(base)
    print(f"Leyendo dataset desde: {source}")
    df = pd.read_csv(source)

    print(f"Filas: {len(df)}, Columnas: {len(df.columns)}")

    target = 'condicion_ocupacion'
    if target not in df.columns:
        raise SystemExit(f"Columna objetivo '{target}' no encontrada en el CSV")

    df = df[df[target].notnull()]

    # Definir columnas esperadas (pueden ajustarse segun dataset)
    numeric_cols = ['edad']
    categorical_cols = [
        'sexo', 'nivel_educativo_cod', 'lengua', 'Etnia',
        'Concentracion', 'Socializar', 'Experiencia', 'Movilidad'
    ]

    # Asegurar que existan en el dataframe
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    X = df[numeric_cols + categorical_cols]
    y = df[target]

    # Pipeline de preprocesado
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ],
        remainder='drop'
    )

    rf = RandomForestClassifier(random_state=42)

    pipe = Pipeline(steps=[('pre', preprocessor), ('clf', rf)])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Grid search CV
    param_grid = {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [5, 10]
    }

    cv = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=2)
    print("Entrenando (esto puede tardar)...")
    cv.fit(X_train, y_train)

    print(f"Mejor params: {cv.best_params_}")
    best = cv.best_estimator_

    # Evaluación
    preds = best.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

    metrics_text = (
        f"Accuracy: {acc:.4f}\n"
        f"Precision (weighted): {prec:.4f}\n"
        f"Recall (weighted): {rec:.4f}\n"
        f"F1 (weighted): {f1:.4f}\n"
    )

    print(metrics_text)
    metrics_path = os.path.join(out_dir, 'metrics_random_forest_sklearn.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(metrics_text)

    # Guardar modelo
    model_path = os.path.join(model_dir, 'rf_model.joblib')
    joblib.dump(best, model_path)
    print(f"Modelo guardado en: {model_path}")

    print("Entrenamiento finalizado.")


if __name__ == '__main__':
    main()
