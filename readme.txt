<<<<<<< HEAD
--------EJECUTAR BASH INTERACTIVO-----

docker compose run --rm spark bash



------LUEGO DENTRO----

pyspark


------------------LIMPIEZA, COLUMNAS Y DATOS IMPORTANTES------------------

from pyspark.sql.functions import col

# Columnas originales que queremos mantener
columnas_mantener = [
    "ANIO", "MES", "CONGLOMERADO", "MUESTRA", "SELVIV", "HOGAR", "REGION", 
    "LLAVE_PANEL", "ESTRATO", "OCUP300", "INGTOT", 
    "C207", "C208", "C366", "C366_1", "C331", "C308_COD", "C309_COD"
]

# Seleccionar solo esas columnas
df = df.select(*columnas_mantener)

# Renombrar las columnas
df = (df
    .withColumnRenamed("ANIO", "anio")
    .withColumnRenamed("MES", "mes")
    .withColumnRenamed("CONGLOMERADO", "conglomerado")
    .withColumnRenamed("MUESTRA", "muestra")
    .withColumnRenamed("SELVIV", "selviv")
    .withColumnRenamed("HOGAR", "hogar")
    .withColumnRenamed("REGION", "region")
    .withColumnRenamed("LLAVE_PANEL", "llave_panel")
    .withColumnRenamed("ESTRATO", "estrato")
    .withColumnRenamed("OCUP300", "ocupado")
    .withColumnRenamed("INGTOT", "ingreso_total")
    .withColumnRenamed("C207", "sexo")
    .withColumnRenamed("C208", "edad")
    .withColumnRenamed("C366", "nivel_educativo")
    .withColumnRenamed("C366_1", "anios_educacion")
    .withColumnRenamed("C331", "horas_trabajadas")
    .withColumnRenamed("C308_COD", "ocupacion")
    .withColumnRenamed("C309_COD", "sector")
)

# Mostrar el resultado
df.printSchema()
df.show(5)
=======
--------EJECUTAR BASH INTERACTIVO-----

docker compose run --rm spark bash



------LUEGO DENTRO----

pyspark


------------------LIMPIEZA, COLUMNAS Y DATOS IMPORTANTES------------------

from pyspark.sql.functions import col

# Columnas originales que queremos mantener
columnas_mantener = [
    "ANIO", "MES", "CONGLOMERADO", "MUESTRA", "SELVIV", "HOGAR", "REGION", 
    "LLAVE_PANEL", "ESTRATO", "OCUP300", "INGTOT", 
    "C207", "C208", "C366", "C366_1", "C331", "C308_COD", "C309_COD"
]

# Seleccionar solo esas columnas
df = df.select(*columnas_mantener)

# Renombrar las columnas
df = (df
    .withColumnRenamed("ANIO", "anio")
    .withColumnRenamed("MES", "mes")
    .withColumnRenamed("CONGLOMERADO", "conglomerado")
    .withColumnRenamed("MUESTRA", "muestra")
    .withColumnRenamed("SELVIV", "selviv")
    .withColumnRenamed("HOGAR", "hogar")
    .withColumnRenamed("REGION", "region")
    .withColumnRenamed("LLAVE_PANEL", "llave_panel")
    .withColumnRenamed("ESTRATO", "estrato")
    .withColumnRenamed("OCUP300", "ocupado")
    .withColumnRenamed("INGTOT", "ingreso_total")
    .withColumnRenamed("C207", "sexo")
    .withColumnRenamed("C208", "edad")
    .withColumnRenamed("C366", "nivel_educativo")
    .withColumnRenamed("C366_1", "anios_educacion")
    .withColumnRenamed("C331", "horas_trabajadas")
    .withColumnRenamed("C308_COD", "ocupacion")
    .withColumnRenamed("C309_COD", "sector")
)

# Mostrar el resultado
df.printSchema()
df.show(5)
>>>>>>> 1b8417f819a151caf548cf5f63bab0903ea8aa3a
