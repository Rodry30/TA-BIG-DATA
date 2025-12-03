from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import os
import glob
import shutil

# --- CONFIGURACIÓN Y SESIÓN ---
spark = SparkSession.builder.appName("ETL: Extraccion, Transformacion y Carga (DW)").getOrCreate()
RUTAS_BASE = "/opt/spark-data/data_warehouse/"
SOURCE_CSV_PATH_1 = "/opt/spark-data/dataset.csv"
SOURCE_CSV_PATH_2 = "/opt/spark-data/dataset1.csv"
SOURCE_CSV_PATH_3 = "/opt/spark-data/dataset2.csv"

# ----------------------------------------------------------------------------------
# FASE 1: EXTRACCIÓN Y LIMPIEZA INICIAL (E)
# ----------------------------------------------------------------------------------
print("\n FASE 1: Extracción y Limpieza Inicial...")

# Leer los 3 CSVs originales con opciones específicas para manejar diferentes formatos
print("Leyendo dataset.csv (con comillas)...")
# dataset.csv tiene comillas dobles, usar opciones por defecto
df1 = spark.read.option("header", "true").option("inferSchema", "true").option("quote", "\"").option("escape", "\"").csv(SOURCE_CSV_PATH_1)

print("Leyendo dataset1.csv (con comillas)...")
# dataset1.csv tiene comillas dobles, usar opciones por defecto
df2 = spark.read.option("header", "true").option("inferSchema", "true").option("quote", "\"").option("escape", "\"").csv(SOURCE_CSV_PATH_2)

print("Leyendo dataset2.csv (sin comillas)...")
# dataset2.csv NO tiene comillas, leer sin especificar quote para que PySpark lo maneje automáticamente
df3 = spark.read.option("header", "true").option("inferSchema", "true").option("multiLine", "false").csv(SOURCE_CSV_PATH_3)

# Obtener todas las columnas comunes y ordenarlas
print("Verificando columnas comunes...")
columnas_comunes = set(df1.columns) & set(df2.columns) & set(df3.columns)
print(f"Columnas comunes encontradas: {len(columnas_comunes)}")

# Seleccionar solo las columnas comunes en el mismo orden para cada DataFrame
columnas_ordenadas = sorted(list(columnas_comunes))
df1_selected = df1.select(*columnas_ordenadas)
df2_selected = df2.select(*columnas_ordenadas)
df3_selected = df3.select(*columnas_ordenadas)

# Unir los 3 datasets (ahora con las mismas columnas en el mismo orden)
print("Uniendo los 3 datasets...")
df = df1_selected.union(df2_selected).union(df3_selected)
print(f"Total de registros después de unir: {df.count()}")

# Columnas que queremos mantener inicialmente
columnas_mantener = ["OCUP300", "C207", "C208", "C366", "C376", "C377", "C375_5", "C375_6", "C359", "C375_1"]

# Verificar que todas las columnas necesarias estén disponibles
columnas_disponibles = set(df.columns)
columnas_faltantes = [col for col in columnas_mantener if col not in columnas_disponibles]
if columnas_faltantes:
    print(f"ADVERTENCIA: Las siguientes columnas no están disponibles: {columnas_faltantes}")
    columnas_mantener = [col for col in columnas_mantener if col in columnas_disponibles]
    print(f"Usando solo las columnas disponibles: {columnas_mantener}")

# Filtrar y renombrar las columnas
print("Filtrando y renombrando columnas...")
df = (df.select(*columnas_mantener)
    .withColumnRenamed("OCUP300", "condicion_ocupacion")
    .withColumnRenamed("C207", "sexo")
    .withColumnRenamed("C208", "edad")
    .withColumnRenamed("C366", "nivel_educativo_cod")
    .withColumnRenamed("C376", "Lengua")
    .withColumnRenamed("C377", "Etnia")
    .withColumnRenamed("C375_5", "Concentracion")
    .withColumnRenamed("C375_6", "Socializar")
    .withColumnRenamed("C375_1", "Movilidad")
    .withColumnRenamed("C359", "Experiencia")
)

# Clasificar condicion_ocupacion: 1 = OCUPADO, resto (0,2,3,4) = DESOCUPADO, mantener nulos
print("Clasificando condicion_ocupacion...")
df = df.withColumn(
    "condicion_ocupacion",
    when(col("condicion_ocupacion") == 1, "OCUPADO")
    .when(col("condicion_ocupacion").isin([0, 2, 3, 4]), "DESOCUPADO")
    .otherwise(col("condicion_ocupacion"))  # Mantiene nulos y otros valores
)
# ----------------------------------------------------------------------------------
# GUARDAR DATASET FILTRADO
# ----------------------------------------------------------------------------------
OUTPUT_DIR = RUTAS_BASE + "dataset_filtrado_temp"
OUTPUT_CSV_PATH = RUTAS_BASE + "dataset_filtrado.csv"
print(f"\n Guardando dataset filtrado...")
# Usar coalesce(1) para generar un solo archivo CSV
df.coalesce(1).write.mode("overwrite").csv(OUTPUT_DIR, header=True)

# Encontrar el archivo part-00000 y renombrarlo a dataset_filtrado.csv
print("Combinando archivos en un solo CSV...")
part_files = glob.glob(os.path.join(OUTPUT_DIR, "part-*.csv"))
if part_files:
    # Mover el archivo part-00000 a dataset_filtrado.csv
    shutil.move(part_files[0], OUTPUT_CSV_PATH)
    # Eliminar el directorio temporal si está vacío
    try:
        os.rmdir(OUTPUT_DIR)
    except:
        pass
else:
    print("Error: No se encontró el archivo generado")

print(f"\n Dataset filtrado guardado exitosamente en: {OUTPUT_CSV_PATH}")
print(f" Total de registros en el dataset filtrado: {df.count()}")
print(f" Columnas finales: {', '.join(df.columns)}")
