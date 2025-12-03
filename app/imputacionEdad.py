from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, desc
import os
from functools import reduce
import operator
import glob
import shutil


def crear_spark():
    return SparkSession.builder.appName("ImputacionPorEdad").getOrCreate()


def map_age_range(df):
    """Añade una columna `age_range` con los rangos solicitados."""
    df = df.withColumn(
        "age_range",
        when(col("edad").isNull(), lit(None))
        .when((col("edad") >= 13) & (col("edad") <= 17), lit("13-17"))
        .when((col("edad") >= 18) & (col("edad") <= 24), lit("18-24"))
        .when((col("edad") >= 25) & (col("edad") <= 34), lit("25-34"))
        .when((col("edad") >= 35) & (col("edad") <= 44), lit("35-44"))
        .when((col("edad") >= 45) & (col("edad") <= 54), lit("45-54"))
        .when((col("edad") >= 55) & (col("edad") <= 64), lit("55-64"))
        .when(col("edad") >= 65, lit("65+"))
        .otherwise(lit(None))
    )
    return df


def eliminar_0_12_con_nulos(df):
    """Elimina filas con edad entre 0 y 12 (inclusive) que tengan ANY valor nulo."""
    cols = df.columns
    # crear condición: existe al menos una columna nula
    null_conditions = [col(c).isNull() for c in cols]
    any_null = reduce(operator.or_, null_conditions)

    # Filtrar fuera las filas con edad 0-12 y any_null == True
    df_clean = df.filter(~(((col("edad") >= 0) & (col("edad") <= 12)) & any_null))
    return df_clean


def calcular_moda_por_grupo(df, column, group_col="age_range"):
    """Devuelve un diccionario {grupo: moda} calculado en Spark para la columna dada."""
    modas = {}
    grupos = [r[0] for r in df.select(group_col).distinct().collect() if r[0] is not None]
    for g in grupos:
        # contar valores no nulos por valor
        agrupado = df.filter((col(group_col) == g) & (col(column).isNotNull()))
        if agrupado.count() == 0:
            modas[g] = None
            continue
        row = agrupado.groupBy(column).count().orderBy(desc("count")).first()
        modas[g] = row[0] if row is not None else None
    return modas


def calcular_moda_global(df, column):
    df_no_nulos = df.filter(col(column).isNotNull())
    if df_no_nulos.count() == 0:
        return None
    row = df_no_nulos.groupBy(column).count().orderBy(desc("count")).first()
    return row[0] if row is not None else None


def imputar_por_rangos(df, group_col="age_range"):
    """Imputa todas las columnas (excepto edad y group_col) por moda dentro de cada rango.
    Si en un rango no existe moda, se usa la moda global como fallback."""
    columnas = [c for c in df.columns if c not in ["edad", group_col]]

    for columna in columnas:
        # comprobar si hay nulos en la columna
        if df.filter(col(columna).isNull()).count() == 0:
            continue

        # calcular modas por grupo y global
        modas_grupo = calcular_moda_por_grupo(df, columna, group_col=group_col)
        moda_global = calcular_moda_global(df, columna)

        # aplicar imputación por cada grupo
        for grupo, moda in modas_grupo.items():
            valor = moda if moda is not None else moda_global
            if valor is None:
                # no hay con qué imputar, saltar
                continue
            df = df.withColumn(
                columna,
                when((col(columna).isNull()) & (col(group_col) == grupo), lit(valor)).otherwise(col(columna))
            )

        # Como seguridad, imputar cualquier nulo restante con moda global (si existe)
        if moda_global is not None:
            df = df.withColumn(columna, when(col(columna).isNull(), lit(moda_global)).otherwise(col(columna)))

    return df


def guardar_csv_unico(df, output_file):
    """Escribe el dataframe a un único CSV con nombre exacto `output_file`.
    Spark escribe por defecto en un directorio con ficheros part-*.csv; aquí
    escribimos en un tmp dir y movemos el part a la ubicación final.
    """
    output_dir = os.path.dirname(output_file)
    tmp_dir = os.path.join(output_dir, "_tmp_imputacion")
    # Asegurar directorio
    os.makedirs(output_dir, exist_ok=True)
    # Escribir en tmp_dir
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(tmp_dir)

    # Buscar el archivo part-*.csv generado por Spark
    part_files = glob.glob(os.path.join(tmp_dir, "part-*.csv"))
    if not part_files:
        raise FileNotFoundError(f"No se encontró ningún part-*.csv en {tmp_dir}")

    part_file = part_files[0]
    # Mover al path final (sobrescribe si existe)
    if os.path.exists(output_file):
        os.remove(output_file)
    shutil.move(part_file, output_file)

    # Limpiar tmp
    shutil.rmtree(tmp_dir)


def encontrar_source_path(base_dir):
    # Posibles ubicaciones (según estructura observada en TR.py)
    candidates = [
        f"{base_dir}/data_warehouse/dataset_filtrado.csv/dataset_filtrado.csv",
        f"{base_dir}/data_warehouse/dataset_filtrado.csv",
        f"{base_dir}/data_warehouse/dataset_filtrado.csv/dataset_filtrado.csv/dataset_filtrado.csv",
        f"{base_dir}/dataset_filtrado.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # si ninguno existe, devolver el primer candidato (Spark puede leer directorios en cluster)
    return candidates[0]


if __name__ == "__main__":
    # Detectar entorno
    if os.path.exists("/opt/spark-data"):
        BASE_DIR = "/opt/spark-data"
        OUTPUT_DIR_DIR = f"{BASE_DIR}/data_warehouse"
        OUTPUT_FILE = os.path.join(OUTPUT_DIR_DIR, "dataset_imputado.csv")
    else:
        BASE_DIR = "data"
        OUTPUT_DIR_DIR = f"{BASE_DIR}/data_warehouse"
        OUTPUT_FILE = os.path.join(OUTPUT_DIR_DIR, "dataset_imputado.csv")

    spark = crear_spark()

    source_path = encontrar_source_path(BASE_DIR)
    print(f"Leyendo dataset desde: {source_path}")
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(source_path)

    total = df.count()
    print(f"Registros leídos: {total}")

    # Agregar rango de edad
    df = map_age_range(df)

    # Eliminar filas 0-12 con nulos
    antes_elim = df.count()
    df = eliminar_0_12_con_nulos(df)
    despues_elim = df.count()
    print(f"Filas eliminadas (0-12 con nulos): {antes_elim - despues_elim}")

    # Imputación por rangos de edad
    print("Aplicando imputación por moda por rango de edad...")
    df = imputar_por_rangos(df, group_col="age_range")

    # Remover columna auxiliar
    if "age_range" in df.columns:
        df = df.drop("age_range")

    # Guardar dataset limpio como un único CSV
    print(f"Guardando dataset imputado en: {OUTPUT_FILE}")
    guardar_csv_unico(df, OUTPUT_FILE)
    print("✅ Imputación y guardado completados.")
