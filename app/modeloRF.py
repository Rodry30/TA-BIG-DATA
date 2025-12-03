from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import os
import glob


def encontrar_source_path(base_dir):
    candidates = [
        f"{base_dir}/data_warehouse/dataset_imputado.csv",
        f"{base_dir}/data_warehouse/dataset_imputado.csv/dataset_imputado.csv",
        f"{base_dir}/data_warehouse/dataset_imputado.csv/part-*.csv",
        f"{base_dir}/dataset_imputado.csv",
    ]
    for p in candidates:
        # if a glob pattern, expand
        if "*" in p:
            matches = glob.glob(p)
            if matches:
                return matches[0]
        else:
            if os.path.exists(p):
                return p
    # fallback: a usual location
    return f"{base_dir}/data_warehouse/dataset_imputado.csv"


def crear_spark():
    return SparkSession.builder.appName("TrainRandomForest").getOrCreate()


def main():
    # Detect environment
    if os.path.exists("/opt/spark-data"):
        BASE_DIR = "/opt/spark-data"
        OUTPUT_DIR = f"{BASE_DIR}/data_warehouse"
        MODEL_DIR = f"{BASE_DIR}/modelos/random_forest_model"
        METRICS_FILE = os.path.join(OUTPUT_DIR, "graficos", "metrics_random_forest.txt")
    else:
        BASE_DIR = "data"
        OUTPUT_DIR = f"{BASE_DIR}/data_warehouse"
        MODEL_DIR = os.path.join("modelos", "random_forest_model")
        METRICS_FILE = os.path.join(OUTPUT_DIR, "graficos", "metrics_random_forest.txt")

    spark = crear_spark()

    source_path = encontrar_source_path(BASE_DIR)
    print(f"Leyendo dataset desde: {source_path}")
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(source_path)

    print(f"Registros: {df.count()}, Columnas: {len(df.columns)}")

    # Target column
    target_col = "condicion_ocupacion"
    if target_col not in df.columns:
        raise SystemExit(f"Columna objetivo '{target_col}' no encontrada en el dataset. Revisa el CSV.")

    # Drop rows with null label
    df = df.filter(col(target_col).isNotNull())

    # Identify numeric and categorical columns
    numeric_types = ("int", "bigint", "double", "float", "long", "decimal")
    numeric_cols = [c for c, t in df.dtypes if (t in numeric_types) and c != target_col]
    categorical_cols = [c for c, t in df.dtypes if (t not in numeric_types) and c != target_col]

    print(f"Numeric cols: {numeric_cols}")
    print(f"Categorical cols: {categorical_cols}")

    # Prepare stages: index label, index & encode categoricals, assemble features
    stages = []

    # Label indexer
    label_indexer = StringIndexer(inputCol=target_col, outputCol="label", handleInvalid='keep')
    stages.append(label_indexer)

    # Categorical indexers and encoders
    indexed_cat_cols = []
    encoded_cat_cols = []
    for c in categorical_cols:
        idx = c + "_idx"
        enc = c + "_enc"
        si = StringIndexer(inputCol=c, outputCol=idx, handleInvalid='keep')
        stages.append(si)
        # OneHotEncoder supports inputCols/outputCols
        indexed_cat_cols.append(idx)
        encoded_cat_cols.append(enc)

    if indexed_cat_cols:
        ohe = OneHotEncoder(inputCols=indexed_cat_cols, outputCols=encoded_cat_cols, handleInvalid='keep')
        stages.append(ohe)

    # Assemble features
    feature_cols = numeric_cols + encoded_cat_cols
    if not feature_cols:
        raise SystemExit("No se encontraron features válidas para entrenamiento.")

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    stages.append(assembler)

    # Define classifier
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42)
    stages.append(rf)

    pipeline = Pipeline(stages=stages)

    # Split
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Train: {train.count()}  Test: {test.count()}")

    # Cross-validation setup
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [20, 50]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()

    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5, parallelism=2)

    print("Entrenando con validación cruzada (esto puede tardar)...")
    cvModel = cv.fit(train)

    print("Evaluando en conjunto de prueba...")
    bestModel = cvModel.bestModel
    predictions = bestModel.transform(test)

    # Metrics
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    precision_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    recall_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    accuracy = acc_eval.evaluate(predictions)
    precision = precision_eval.evaluate(predictions)
    recall = recall_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)

    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision (weighted): {precision:.4f}\n"
        f"Recall (weighted): {recall:.4f}\n"
        f"F1 (weighted): {f1:.4f}\n"
    )

    print("\n=== METRICAS EN TEST ===")
    print(metrics_text)

    # Guardar métricas
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write(metrics_text)

    # Guardar modelo
    print(f"Guardando modelo en: {MODEL_DIR}")
    try:
        # Remove existing model dir if exists
        if os.path.exists(MODEL_DIR):
            import shutil
            shutil.rmtree(MODEL_DIR)
        bestModel.write().overwrite().save(MODEL_DIR)
    except Exception as e:
        print(f"Advertencia: no se pudo guardar el modelo en filesystem local: {e}")

    print("Entrenamiento completado.")


if __name__ == '__main__':
    main()
