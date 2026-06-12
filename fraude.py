import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.ensemble import IsolationForest

# Configuración de rutas locales profesionales
DATA_PROCESSED_DIR = "data/processed"

def extract_ieee_data():
    """
    EXTRACT: Descarga y une las tablas de transacciones e identidad.
    """
    print("\n[ETL - EXTRACT] Iniciando verificación del dataset...")
    download_path = kagglehub.dataset_download("lnasiri007/ieeecis-fraud-detection")
    files_in_download = os.listdir(download_path)
    
    transaction_file = next((f for f in files_in_download if 'transaction' in f.lower() and 'train' in f.lower()), None)
    identity_file = next((f for f in files_in_download if 'identity' in f.lower() and 'train' in f.lower()), None)
    
    tx_path = os.path.join(download_path, transaction_file)
    id_path = os.path.join(download_path, identity_file)
    
    print("[ETL - EXTRACT] Cargando datos en Pandas...")
    df_transaction = pd.read_csv(tx_path)
    df_identity = pd.read_csv(id_path)
    
    print("[ETL - EXTRACT] Realizando Left Join mediante 'TransactionID'...")
    df_merged = pd.merge(df_transaction, df_identity, on='TransactionID', how='left')
    return df_merged


def transform_and_enrich(df):
    """
    TRANSFORM: Limpieza, preparación de variables esenciales y 
    aplicación de la Capa 1 No Supervisada (Isolation Forest).
    """
    print("\n[ETL - TRANSFORM] Iniciando fase de transformación...")
    df_transformed = df.copy()
    
    # 1. Selección de variables numéricas clave para el Isolation Forest
    # Elegimos variables financieras y de comportamiento del cliente básico
    features_for_anomaly = ['TransactionAmt', 'TransactionDT', 'card1', 'card2', 'card3', 'card5']
    
    print(f"[ETL - TRANSFORM] Seleccionando variables para Isolation Forest: {features_for_anomaly}")
    
    # Creamos un sub-dataframe de trabajo y manejamos los nulos con un valor bandera
    df_sub = df_transformed[features_for_anomaly].copy()
    for col in features_for_anomaly:
        if df_sub[col].isnull().sum() > 0:
            df_sub[col] = df_sub[col].fillna(-999) # Indicador explícito de dato faltante
            
    # 2. Configuración y entrenamiento de Isolation Forest
    # El dataset oficial tiene cerca de un 3.5% de fraude real, fijamos la contaminación ahí
    print("[ETL - TRANSFORM] Entrenando Isolation Forest (Capa No Supervisada)...")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.035, random_state=42, n_jobs=-1)
    
    # Predecimos anomalías (-1 = Anómalo, 1 = Normal)
    preds = iso_forest.fit_predict(df_sub)
    
    # Mapeamos a binario estándar de negocio: 1 para anómalo, 0 para normal
    df_transformed['is_anomaly'] = [1 if p == -1 else 0 for p in preds]
    # Guardamos el score de anomalía (valores más bajos/negativos indican mayor rareza)
    df_transformed['anomaly_score'] = iso_forest.score_samples(df_sub)
    
    print("[ETL - TRANSFORM] Transformación y etiquetado de anomalías finalizado.")
    return df_transformed


def load_processed_data(df, output_filename="enriched_transactions.csv"):
    """
    LOAD: Almacena el dataset enriquecido listo para el modelo supervisado o analistas.
    """
    print(f"\n[ETL - LOAD] Guardando datos procesados en la ruta local...")
    if not os.path.exists(DATA_PROCESSED_DIR):
        os.makedirs(DATA_PROCESSED_DIR)
        
    output_path = os.path.join(DATA_PROCESSED_DIR, output_filename)
    
    # Guardamos en formato CSV comprimido o estándar
    # Debido a las 436 columnas, esto puede tomar un par de minutos en el disco
    df.to_csv(output_path, index=False)
    print(f"[ETL - LOAD] ARCHIVO GUARDADO EXITOSAMENTE EN: {output_path}")


# Orquestador del Pipeline ETL
if __name__ == "__main__":
    print("==================================================================")
    print("=== RUNNING PREVENCION DE FRAUDE PIPELINE: PRODUCTION VERSION ===")
    print("==================================================================")
    
    # Ejecución secuencial
    df_raw = extract_ieee_data()
    df_enriched = transform_and_enrich(df_raw)
    load_processed_data(df_enriched)
    
    print("\n=== Control de calidad del Pipeline ===")
    print("Distribución de anomalías encontradas por Isolation Forest:")
    print(df_enriched['is_anomaly'].value_counts())
    print("==================================================================")