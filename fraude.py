import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.ensemble import IsolationForest
# Importamos el cargador centralizado para respetar la inyección de dependencias
from generar_eda import cargar_configuracion

def extract_ieee_data(config):
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


def transform_and_enrich(df, config):
    """
    TRANSFORM: Limpieza, preparación de variables esenciales y 
    aplicación de la Capa 1 No Supervisada (Isolation Forest) gobernada por YAML.
    """
    print("\n[ETL - TRANSFORM] Iniciando fase de transformación...")
    df_transformed = df.copy()
    
    # CORRECCIÓN: Acceso directo a la llave raíz del config.yaml real
    iso_config = config['isolation_forest']
    
    # Respaldo seguro en caso de que no se listen las variables explícitamente en el YAML
    features_for_anomaly = iso_config['features'] if 'features' in iso_config else ['TransactionAmt', 'TransactionDT', 'card1', 'card2', 'card3', 'card5']
    
    print(f"[ETL - TRANSFORM] Seleccionando variables para Isolation Forest desde config: {features_for_anomaly}")
    
    # Creamos un sub-dataframe de trabajo y manejamos los nulos con un valor bandera
    df_sub = df_transformed[features_for_anomaly].copy()
    for col in features_for_anomaly:
        if df_sub[col].isnull().sum() > 0:
            df_sub[col] = df_sub[col].fillna(-999) # Indicador explícito de dato faltante
            
    # Configuración dinámicamente inyectada
    print("[ETL - TRANSFORM] Entrenando Isolation Forest (Capa No Supervisada)...")
    iso_forest = IsolationForest(
        n_estimators=iso_config['n_estimators'], 
        contamination=iso_config['contamination'], 
        random_state=iso_config['random_state'], 
        n_jobs=-1
    )
    
    # Predecimos anomalías (-1 = Anómalo, 1 = Normal)
    preds = iso_forest.fit_predict(df_sub)
    
    # Mapeamos a binario estándar de negocio: 1 para anómalo, 0 para normal
    df_transformed['is_anomaly'] = [1 if p == -1 else 0 for p in preds]
    # Guardamos el score de anomalía
    df_transformed['anomaly_score'] = iso_forest.score_samples(df_sub)
    
    print("[ETL - TRANSFORM] Transformación y etiquetado de anomalías finalizado.")
    return df_transformed


def load_processed_data(df, config, output_filename="enriched_transactions.csv"):
    """
    LOAD: Almacena el dataset enriquecido respetando las rutas del config.yaml.
    """
    # CORRECCIÓN: Ajuste de la sección 'rutas' a la llave estructural real 'data'
    processed_dir = config['data']['processed_dir']
    print(f"\n[ETL - LOAD] Guardando datos procesados en la ruta configurada...")
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    output_path = os.path.join(processed_dir, output_filename)
    
    # Guardamos en formato CSV
    df.to_csv(output_path, index=False)
    print(f"[ETL - LOAD] ARCHIVO GUARDADO EXITOSAMENTE EN: {output_path}")


# Orquestador del Pipeline ETL
if __name__ == "__main__":
    print("==================================================================")
    print("=== RUNNING PREVENCION DE FRAUDE PIPELINE: PRODUCTION VERSION ===")
    print("==================================================================")
    
    # Cargar la configuración centralizada antes de iniciar
    config = cargar_configuracion()
    
    # Ejecución secuencial parametrizada
    df_raw = extract_ieee_data(config)
    df_enriched = transform_and_enrich(df_raw, config)
    load_processed_data(df_enriched, config)
    
    print("\n=== Control de calidad del Pipeline ===")
    print("Distribución de anomalías encontradas por Isolation Forest:")
    print(df_enriched['is_anomaly'].value_counts())
    print("==================================================================")