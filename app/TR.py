from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, desc, isnan, isnull, count, sum as spark_sum
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
import os

spark = SparkSession.builder.appName("ETL: Transformación e Inserción").getOrCreate()

# --- FUNCIONES DE IMPUTACIÓN ---
def imputar_estrato(df):
    """Imputa los nulos de estrato con la mediana global"""
    mediana = df.approxQuantile("estrato", [0.5], 0.0)[0]
    df = df.withColumn("estrato", when(col("estrato").isNull(), lit(mediana)).otherwise(col("estrato")))
    return df

def imputar_condicion_ocupacion(df):
    """Imputa los nulos de condicion_ocupacion con la moda global"""
    # Filtrar nulos para obtener la moda
    df_no_nulos = df.filter(col("condicion_ocupacion").isNotNull())
    if df_no_nulos.count() > 0:
        moda = df_no_nulos.groupBy("condicion_ocupacion").count().orderBy(desc("count")).first()[0]
        df = df.withColumn("condicion_ocupacion", when(col("condicion_ocupacion").isNull(), lit(moda)).otherwise(col("condicion_ocupacion")))
    return df

def imputar_ingreso_total(df):
    """Imputa ingreso_total solo para ocupados con edad >= 13"""
    grupos = df.select("sexo", "estrato").distinct().collect()
    for g in grupos:
        sexo_val = g["sexo"]
        estrato_val = g["estrato"]
        # Calcular mediana solo para filas con ingreso no nulo, edad >= 13 y ocupados
        mediana_grupo = df.filter(
            (col("sexo") == sexo_val) & 
            (col("estrato") == estrato_val) & 
            (col("ingreso_total").isNotNull()) &
            (col("edad") >= 13) &
            (col("condicion_ocupacion") == 1)
        ).approxQuantile("ingreso_total", [0.5], 0.0)[0]
        
        df = df.withColumn(
            "ingreso_total",
            when(
                (col("ingreso_total").isNull()) &
                (col("sexo") == sexo_val) &
                (col("estrato") == estrato_val) &
                (col("edad") >= 13) &
                (col("condicion_ocupacion") == 1),
                lit(mediana_grupo)
            ).otherwise(col("ingreso_total"))
        )
    
    # Para desocupados o edades < 13, forzar ingreso_total a NULL
    df = df.withColumn(
        "ingreso_total",
        when((col("edad") < 13) | (col("condicion_ocupacion") != 1), None).otherwise(col("ingreso_total"))
    )
    
    return df

def imputar_predictores(df):
    """Imputa nivel_educativo_cod y anios_educacion con la moda global"""
    columnas = ["nivel_educativo_cod", "anios_educacion"]
    for c in columnas:
        if c in df.columns:
            df_no_nulos = df.filter(col(c).isNotNull())
            if df_no_nulos.count() > 0:
                moda_global = df_no_nulos.groupBy(c).count().orderBy(desc("count")).first()[0]
                df = df.withColumn(c, when(col(c).isNull(), lit(moda_global)).otherwise(col(c)))
    return df

def map_ocupacion(df):
    """Convierte los valores de condicion_ocupacion a solo Ocupado / Desocupado"""
    df = df.withColumn(
        "condicion_ocupacion",
        when(col("condicion_ocupacion") == 1, "Ocupado").otherwise("Desocupado")
    )
    return df

def fase2_imputacion(df):
    """Aplica todas las imputaciones disponibles según las columnas presentes"""
    columnas_disponibles = set(df.columns)
    
    # Solo aplicar imputaciones si las columnas existen
    if "estrato" in columnas_disponibles:
        df = imputar_estrato(df)
    
    if "condicion_ocupacion" in columnas_disponibles:
        # Verificar si ya está mapeado a OCUPADO/DESOCUPADO o necesita mapeo
        valores_unicos = [row[0] for row in df.select("condicion_ocupacion").distinct().collect() if row[0] is not None]
        if any(v in [0, 1, 2, 3, 4] for v in valores_unicos if isinstance(v, (int, float))):
            # Necesita mapeo
            df = map_ocupacion(df)
        # Imputar nulos si existen
        if df.filter(col("condicion_ocupacion").isNull()).count() > 0:
            df = imputar_condicion_ocupacion(df)
    
    if "ingreso_total" in columnas_disponibles:
        df = imputar_ingreso_total(df)
    
    if "nivel_educativo_cod" in columnas_disponibles or "anios_educacion" in columnas_disponibles:
        df = imputar_predictores(df)
    
    return df

# --- FUNCIONES DE ANÁLISIS Y VISUALIZACIÓN ---
def crear_cluster_edad(df):
    """Crea clusters de edad para análisis"""
    df = df.withColumn(
        "cluster_edad",
        when(col("edad").isNull(), "Sin Edad")
        .when(col("edad") < 5, "0-4 años")
        .when(col("edad") < 13, "5-12 años")
        .when(col("edad") < 18, "13-17 años")
        .when(col("edad") < 25, "18-24 años")
        .when(col("edad") < 35, "25-34 años")
        .when(col("edad") < 45, "35-44 años")
        .when(col("edad") < 55, "45-54 años")
        .when(col("edad") < 65, "55-64 años")
        .otherwise("65+ años")
    )
    return df

def analizar_nulos_por_columna(df, output_dir="/opt/spark-data/data_warehouse/graficos/"):
    """Analiza nulos por columna agrupados por cluster de edad y genera gráficos"""
    print("\n" + "="*60)
    print("ANÁLISIS DE NULOS POR COLUMNA (CLUSTERING POR EDAD)")
    print("="*60)
    
    # Crear cluster de edad
    df_clustered = crear_cluster_edad(df)
    
    # Convertir a Pandas para visualización (muestra representativa si es muy grande)
    print("Convirtiendo a Pandas para análisis...")
    total_registros = df_clustered.count()
    if total_registros > 50000:
        print(f"Dataset grande ({total_registros} registros). Usando muestra de 50000 registros...")
        df_pandas = df_clustered.sample(False, 50000/total_registros, seed=42).toPandas()
    else:
        df_pandas = df_clustered.toPandas()
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Análisis de nulos por columna y cluster de edad
    columnas_analizar = [c for c in df_pandas.columns if c != "cluster_edad"]
    
    # Crear DataFrame de resumen de nulos
    resumen_nulos = []
    for columna in columnas_analizar:
        for cluster in df_pandas["cluster_edad"].unique():
            mask = (df_pandas["cluster_edad"] == cluster)
            total = mask.sum()
            nulos = df_pandas.loc[mask, columna].isnull().sum()
            porcentaje = (nulos / total * 100) if total > 0 else 0
            resumen_nulos.append({
                "columna": columna,
                "cluster_edad": cluster,
                "total_registros": total,
                "nulos": nulos,
                "porcentaje_nulos": porcentaje
            })
    
    df_resumen = pd.DataFrame(resumen_nulos)
    
    # 2. Gráfico de nulos por columna y cluster
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Gráfico 1: Heatmap de porcentaje de nulos
    pivot_nulos = df_resumen.pivot(index="columna", columns="cluster_edad", values="porcentaje_nulos")
    sns.heatmap(pivot_nulos, annot=True, fmt=".1f", cmap="Reds", ax=axes[0], cbar_kws={'label': '% Nulos'})
    axes[0].set_title("Porcentaje de Valores Nulos por Columna y Cluster de Edad", fontsize=16, fontweight='bold')
    axes[0].set_xlabel("Cluster de Edad", fontsize=14)
    axes[0].set_ylabel("Columna", fontsize=14)
    axes[0].tick_params(axis='x', rotation=45, labelsize=12)
    axes[0].tick_params(axis='y', rotation=0, labelsize=12)
    
    # Gráfico 2: Barras apiladas de nulos por cluster
    pivot_count = df_resumen.pivot(index="columna", columns="cluster_edad", values="nulos")
    pivot_count.plot(kind='bar', stacked=True, ax=axes[1], colormap='Set3')
    axes[1].set_title("Cantidad de Valores Nulos por Columna y Cluster de Edad", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Columna", fontsize=12)
    axes[1].set_ylabel("Cantidad de Nulos", fontsize=12)
    axes[1].legend(title="Cluster de Edad", bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "analisis_nulos_por_edad.png"), dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico de nulos guardado en: {os.path.join(output_dir, 'analisis_nulos_por_edad.png')}")
    plt.close()
    
    # 3. Mostrar resumen en consola
    print("\n--- RESUMEN DE NULOS POR CLUSTER DE EDAD ---")
    for cluster in sorted(df_resumen["cluster_edad"].unique()):
        print(f"\n📊 Cluster: {cluster}")
        cluster_data = df_resumen[df_resumen["cluster_edad"] == cluster].sort_values("porcentaje_nulos", ascending=False)
        for _, row in cluster_data.head(10).iterrows():
            if row["nulos"] > 0:
                print(f"   {row['columna']}: {row['nulos']} nulos ({row['porcentaje_nulos']:.1f}%)")
    
    return df_resumen

def analizar_correlacion_ocupacion(df, output_dir="/opt/spark-data/data_warehouse/graficos/"):
    """Analiza correlación entre ocupación, edad y demás variables"""
    print("\n" + "="*60)
    print("ANÁLISIS DE CORRELACIÓN CON CONDICIÓN DE OCUPACIÓN")
    print("="*60)
    
    # Convertir a Pandas para análisis (muestra si es muy grande)
    print("Convirtiendo a Pandas para análisis de correlación...")
    total_registros = df.count()
    if total_registros > 50000:
        print(f"Dataset grande ({total_registros} registros). Usando muestra de 50000 registros...")
        df_pandas = df.sample(False, 50000/total_registros, seed=42).toPandas()
    else:
        df_pandas = df.toPandas()
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Preparar datos numéricos
    # Convertir condicion_ocupacion a numérico (OCUPADO=1, DESOCUPADO=0, nulos se mantienen)
    df_pandas["condicion_ocupacion_num"] = df_pandas["condicion_ocupacion"].map({
        "OCUPADO": 1,
        "DESOCUPADO": 0
    })

    # Invertir variable Experiencia si existe (1=Si, 2=No) -> (1=Si, 0=No)
    if "Experiencia" in df_pandas.columns:
        print("Invertiendo variable Experiencia (1=Si, 2=No -> 1=Si, 0=No)...")
        # Mapear 1 -> 1, 2 -> 0. Cualquier otro valor se mantiene (o se convierte a NaN si no coincide)
        # Asumiendo que los valores son numéricos 1 y 2
        df_pandas["Experiencia"] = df_pandas["Experiencia"].map({1: 1, 2: 0})
    
    # Seleccionar columnas numéricas para correlación
    columnas_numericas = df_pandas.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filtrar columnas relevantes (excluir IDs y columnas no relevantes)
    columnas_relevantes = [c for c in columnas_numericas if c in ["edad", "sexo", "nivel_educativo_cod", 
                                                                  "Lengua", "Etnia", "Concentracion", 
                                                                  "Socializar", "Experiencia", "Movilidad",
                                                                  "condicion_ocupacion_num"]]
    
    df_corr = df_pandas[columnas_relevantes]
    
    # Calcular matriz de correlación
    correlation_matrix = df_corr.corr()
    
    # 1. Gráfico de matriz de correlación completa
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Gráfico 1: Heatmap de correlación completo
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[0, 0])
    axes[0, 0].set_title("Matriz de Correlación Completa", fontsize=14, fontweight='bold')
    
    # Gráfico 2: Correlación con condicion_ocupacion
    if "condicion_ocupacion_num" in correlation_matrix.columns:
        target_corr = correlation_matrix["condicion_ocupacion_num"].drop("condicion_ocupacion_num")
        target_corr_sorted = target_corr.sort_values(ascending=False)
        
        colors = ['green' if x > 0 else 'red' for x in target_corr_sorted.values]
        axes[0, 1].barh(range(len(target_corr_sorted)), target_corr_sorted.values, color=colors)
        axes[0, 1].set_yticks(range(len(target_corr_sorted)))
        axes[0, 1].set_yticklabels(target_corr_sorted.index)
        axes[0, 1].set_xlabel("Correlación con Condición de Ocupación", fontsize=12)
        axes[0, 1].set_title("Correlación de Variables con Condición de Ocupación", fontsize=14, fontweight='bold')
        axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        
        # Agregar valores en las barras
        for i, v in enumerate(target_corr_sorted.values):
            axes[0, 1].text(v, i, f' {v:.3f}', va='center', fontsize=9)
    
    # Gráfico 3: Distribución de ocupación por edad
    df_pandas_ocupados = df_pandas[df_pandas["condicion_ocupacion"].isin(["OCUPADO", "DESOCUPADO"])]
    if not df_pandas_ocupados.empty:
        # Crear bins de edad
        bins = [0, 5, 13, 18, 25, 35, 45, 55, 65, 100]
        labels = ["0-4", "5-12", "13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        df_pandas_ocupados["rango_edad"] = pd.cut(df_pandas_ocupados["edad"], bins=bins, labels=labels, right=False)
        
        ocupacion_por_edad = pd.crosstab(df_pandas_ocupados["rango_edad"], 
                                         df_pandas_ocupados["condicion_ocupacion"], 
                                         normalize='index') * 100
        
        ocupacion_por_edad.plot(kind='bar', stacked=True, ax=axes[1, 0], colormap='RdYlGn')
        axes[1, 0].set_title("Distribución de Ocupación por Rango de Edad (%)", fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel("Rango de Edad", fontsize=12)
        axes[1, 0].set_ylabel("Porcentaje", fontsize=12)
        axes[1, 0].legend(title="Condición", bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Gráfico 4: Scatter plot edad vs ocupación (con jitter)
    if not df_pandas_ocupados.empty and "edad" in df_pandas_ocupados.columns:
        # Agregar jitter para mejor visualización
        np.random.seed(42)
        jitter = np.random.normal(0, 0.1, len(df_pandas_ocupados))
        df_pandas_ocupados["edad_jitter"] = df_pandas_ocupados["condicion_ocupacion_num"] + jitter
        
        # Muestra aleatoria para no sobrecargar el gráfico
        muestra = df_pandas_ocupados.sample(min(5000, len(df_pandas_ocupados)), random_state=42)
        
        axes[1, 1].scatter(muestra["edad"], muestra["edad_jitter"], 
                          alpha=0.3, s=10, c=muestra["condicion_ocupacion_num"], 
                          cmap='RdYlGn', edgecolors='none')
        axes[1, 1].set_xlabel("Edad", fontsize=12)
        axes[1, 1].set_ylabel("Condición de Ocupación (0=Desocupado, 1=Ocupado)", fontsize=12)
        axes[1, 1].set_title("Relación Edad vs Condición de Ocupación", fontsize=14, fontweight='bold')
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_yticklabels(["DESOCUPADO", "OCUPADO"])
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "analisis_correlacion_ocupacion.png"), dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico de correlación guardado en: {os.path.join(output_dir, 'analisis_correlacion_ocupacion.png')}")
    plt.close()
    
    # Mostrar correlaciones en consola
    if "condicion_ocupacion_num" in correlation_matrix.columns:
        print("\n--- CORRELACIONES CON CONDICIÓN DE OCUPACIÓN ---")
        target_corr = correlation_matrix["condicion_ocupacion_num"].drop("condicion_ocupacion_num")
        target_corr_sorted = target_corr.sort_values(ascending=False)
        for var, corr in target_corr_sorted.items():
            print(f"   {var}: {corr:.4f}")
    
    return correlation_matrix

def realizar_clustering_y_analisis_nulos(df, output_dir="/opt/spark-data/data_warehouse/graficos/"):
    """
    Realiza análisis de nulos y clustering (K-Means) excluyendo condicion_ocupacion.
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE CLUSTERING Y NULOS (EXCLUYENDO OCUPACIÓN)")
    print("="*60)
    
    # 1. Selección de variables (Excluyendo condicion_ocupacion y sexo si se desea, aquí excluimos ocupación)
    # Variables potenciales según dataset: edad, nivel_educativo_cod, Lengua, Etnia, Concentracion, Socializar, Experiencia, Movilidad
    features_clustering = ["edad", "nivel_educativo_cod", "Lengua", "Etnia", "Concentracion", "Socializar", "Experiencia", "Movilidad"]
    
    # Verificar que existan en el DF
    features_existentes = [c for c in features_clustering if c in df.columns]
    print(f"Variables seleccionadas para clustering: {features_existentes}")
    
    if len(features_existentes) < 2:
        print("⚠️ No hay suficientes variables para realizar clustering.")
        return
    
    # 2. Análisis de % de nulos de cada columna rescatada
    print("\n--- PORCENTAJE DE NULOS POR VARIABLE SELECCIONADA ---")
    total_count = df.count()
    df_nulos = df.select([
        (count(when(col(c).isNull() | isnan(c), c)) / total_count * 100).alias(c)
        for c in features_existentes
    ])
    
    # Mostrar resultados
    df_nulos.show()
    
    # Guardar reporte de nulos
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "reporte_nulos_clustering.txt"), "w") as f:
        f.write("PORCENTAJE DE NULOS POR VARIABLE SELECCIONADA\n")
        f.write("=============================================\n")
        row = df_nulos.collect()[0]
        for c in features_existentes:
            val = row[c]
            f.write(f"{c}: {val:.2f}%\n")
            print(f"{c}: {val:.2f}%")

    # 3. Preparación para Clustering
    # Eliminar filas con nulos en las features seleccionadas para poder ejecutar K-Means
    df_clean = df.na.drop(subset=features_existentes)
    count_clean = df_clean.count()
    print(f"\nRegistros válidos para clustering (sin nulos en features): {count_clean} (Original: {total_count})")
    
    if count_clean == 0:
        print("⚠️ No quedan registros después de eliminar nulos. No se puede realizar clustering.")
        return

    # Vector Assembler
    assembler = VectorAssembler(inputCols=features_existentes, outputCol="features")
    df_assembled = assembler.transform(df_clean)
    
    # Escalar datos (StandardScaler)
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    scalerModel = scaler.fit(df_assembled)
    df_scaled = scalerModel.transform(df_assembled)
    
    # 4. Aplicar K-Means
    k = 3 # Número de clusters arbitrario, podría optimizarse con Elbow Method
    kmeans = KMeans(featuresCol="scaledFeatures", k=k, seed=42)
    model = kmeans.fit(df_scaled)
    
    # Agregar predicciones
    predictions = model.transform(df_scaled)
    
    # Evaluar clusters (Centros)
    centers = model.clusterCenters()
    print("\nCentros de los Clusters:")
    for i, center in enumerate(centers):
        print(f"Cluster {i}: {center}")
        
    # 5. Visualización (PCA 2D)
    print("\nGenerando visualización de clusters (PCA)...")
    
    # Reducir a 2 componentes principales
    pca = PCA(k=2, inputCol="scaledFeatures", outputCol="pcaFeatures")
    pcaModel = pca.fit(predictions)
    result_pca = pcaModel.transform(predictions)
    
    # Convertir a Pandas para graficar
    # Tomar muestra si es muy grande
    if count_clean > 10000:
        df_viz = result_pca.select("pcaFeatures", "prediction").sample(False, 10000/count_clean, seed=42).toPandas()
    else:
        df_viz = result_pca.select("pcaFeatures", "prediction").toPandas()
        
    # Extraer componentes x, y
    df_viz["pca_x"] = df_viz["pcaFeatures"].apply(lambda v: v[0])
    df_viz["pca_y"] = df_viz["pcaFeatures"].apply(lambda v: v[1])
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="pca_x", y="pca_y",
        hue="prediction",
        palette="viridis",
        data=df_viz,
        alpha=0.6,
        s=50
    )
    plt.title(f"Visualización de Clusters (K-Means k={k}) - PCA 2D", fontsize=16)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend(title="Cluster")
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, "analisis_clustering.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico de clusters guardado en: {output_path}")
    plt.close()
    
    # 6. Interpretación de Clusters
    print("\n--- INTERPRETACIÓN DE CLUSTERS ---")
    
    # Calcular medias globales de las features (usando Pandas para facilitar)
    # Convertir features a columnas individuales si es necesario, pero ya tenemos df_viz con pca
    # Mejor usar df_clean original para calcular medias reales
    
    # Tomar muestra para interpretación si es muy grande
    if count_clean > 50000:
        df_interp = df_clean.select(features_existentes).sample(False, 50000/count_clean, seed=42).toPandas()
    else:
        df_interp = df_clean.select(features_existentes).toPandas()
        
    global_means = df_interp.mean()
    
    # Agregar predicciones al dataframe de pandas
    # Nota: predictions es Spark DF, df_interp es Pandas. 
    # Para hacerlo bien, convertimos predictions a Pandas (ya lo hicimos parcialmente en df_viz pero solo PCA)
    
    # Vamos a usar 'predictions' de Spark para agrupar y calcular medias por cluster
    cluster_means_spark = predictions.groupBy("prediction").mean(*features_existentes).collect()
    
    interpretation_file = os.path.join(output_dir, "interpretacion_clusters.txt")
    with open(interpretation_file, "w", encoding="utf-8") as f:
        f.write("INTERPRETACIÓN DE CLUSTERS (K-Means)\n")
        f.write("====================================\n\n")
        
        for row in sorted(cluster_means_spark, key=lambda x: x["prediction"]):
            cluster_id = row["prediction"]
            f.write(f"Cluster {cluster_id}:\n")
            print(f"\nCluster {cluster_id}:")
            
            descriptions = []
            for feature in features_existentes:
                cluster_val = row[f"avg({feature})"]
                global_val = global_means[feature]
                
                # Calcular desviación porcentual
                if global_val != 0:
                    diff_pct = ((cluster_val - global_val) / abs(global_val)) * 100
                else:
                    diff_pct = 0
                
                # Definir umbral para considerar significativo (e.g., 10%)
                if diff_pct > 10:
                    desc = f"{feature} ALTO (+{diff_pct:.1f}%)"
                    descriptions.append(desc)
                elif diff_pct < -10:
                    desc = f"{feature} BAJO ({diff_pct:.1f}%)"
                    descriptions.append(desc)
            
            if not descriptions:
                desc_str = "Valores promedio en todas las variables."
            else:
                desc_str = ", ".join(descriptions)
            
            f.write(f"  Perfil: {desc_str}\n")
            f.write(f"  Detalles:\n")
            print(f"  Perfil: {desc_str}")
            
            for feature in features_existentes:
                cluster_val = row[f"avg({feature})"]
                f.write(f"    - {feature}: {cluster_val:.2f} (Global: {global_means[feature]:.2f})\n")
                
        print(f"\n✅ Interpretación guardada en: {interpretation_file}")

    return predictions

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    # Detectar entorno (Docker vs Local)
    if os.path.exists("/opt/spark-data"):
        print("🔧 Entorno detectado: Docker")
        BASE_DIR = "/opt/spark-data"
        # En Linux/Docker, Spark lee directorios sin problemas
        SOURCE_CSV_PATH = f"{BASE_DIR}/data_warehouse/dataset_filtrado.csv/dataset_filtrado.csv"
        OUTPUT_CSV_PATH = f"{BASE_DIR}/data_warehouse/dataset_limpio"
        GRAFICOS_DIR = f"{BASE_DIR}/data_warehouse/graficos/"
    else:
        print("🔧 Entorno detectado: Local (Windows/Other)")
        BASE_DIR = "data"
        # En Windows, a veces es necesario apuntar al archivo específico si hay problemas con winutils
        SOURCE_CSV_PATH = f"{BASE_DIR}/data_warehouse/dataset_filtrado.csv/dataset_filtrado.csv/dataset_filtrado.csv"
        OUTPUT_CSV_PATH = f"{BASE_DIR}/data_warehouse/dataset_limpio"
        GRAFICOS_DIR = f"{BASE_DIR}/data_warehouse/graficos/"
    
    # Leer dataset filtrado
    print("Leyendo dataset filtrado...")
    # Spark puede leer tanto archivos como directorios
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(SOURCE_CSV_PATH)
    
    print(f"\nTotal de registros: {df.count()}")
    print(f"Columnas disponibles: {', '.join(df.columns)}")
    
    # ANÁLISIS ANTES DE IMPUTACIÓN
    print("\n" + "="*60)
    print("ANÁLISIS INICIAL (ANTES DE IMPUTACIÓN)")
    print("="*60)
    
    # 1. Análisis de nulos por columna con clustering por edad
    resumen_nulos = analizar_nulos_por_columna(df, output_dir=GRAFICOS_DIR)
    
    # 2. Análisis de correlación con ocupación
    matriz_correlacion = analizar_correlacion_ocupacion(df, output_dir=GRAFICOS_DIR)

    # 3. Análisis de Clustering y Nulos (Nueva funcionalidad)
    df_clusters = realizar_clustering_y_analisis_nulos(df, output_dir=GRAFICOS_DIR)
    
    # IMPUTACIÓN
    print("\n" + "="*60)
    print("APLICANDO IMPUTACIÓN")
    print("="*60)
    
    print("\nAntes de la imputación:")
    # Mostrar algunas columnas si existen
    columnas_mostrar = [c for c in ["estrato", "condicion_ocupacion", "ingreso_total", 
                                     "nivel_educativo_cod", "anios_educacion"] if c in df.columns]
    if columnas_mostrar:
        df.select(*columnas_mostrar).show(5)
    
    # IMPUTACIÓN (DESACTIVADA POR SOLICITUD DEL USUARIO)
    print("\n" + "="*60)
    print("IMPUTACIÓN OMITIDA (ANÁLISIS DE DATOS ORIGINALES)")
    print("="*60)
    
    # print("\nAntes de la imputación:")
    # # Mostrar algunas columnas si existen
    # columnas_mostrar = [c for c in ["estrato", "condicion_ocupacion", "ingreso_total", 
    #                                  "nivel_educativo_cod", "anios_educacion"] if c in df.columns]
    # if columnas_mostrar:
    #     df.select(*columnas_mostrar).show(5)
    
    # # Aplicar imputación y mapeo de ocupacion
    # # df = fase2_imputacion(df)
    
    # print("\nDespués de la imputación y mapeo Ocupado/Desocupado:")
    # if columnas_mostrar:
    #     df.select(*columnas_mostrar).show(5)
    
    # # Guardar dataset limpio
    # print(f"\nGuardando dataset limpio en: {OUTPUT_CSV_PATH}")
    # # df.coalesce(1).write.mode("overwrite").csv(OUTPUT_CSV_PATH, header=True)
    # print(f"✅ Dataset limpio guardado exitosamente")
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
    print(f"✅ Gráficos guardados en: {GRAFICOS_DIR}")
    print(f"   - analisis_nulos_por_edad.png")
    print(f"   - analisis_correlacion_ocupacion.png")
    print(f"   - analisis_clustering.png")
