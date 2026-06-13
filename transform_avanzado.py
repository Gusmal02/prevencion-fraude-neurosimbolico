import os
import pandas as pd
import numpy as np
# Respetamos el principio de Inyección de Dependencias
from generar_eda import cargar_configuracion

def aplicar_feature_engineering_avanzado(config):
    # 1. Recuperar rutas dinámicamente desde la sección 'data' del config.yaml
    processed_dir = config['data']['processed_dir']
    
    input_file = os.path.join(processed_dir, "enriched_transactions.csv")
    output_file = os.path.join(processed_dir, "final_features_transactions.csv")
    
    print(f"[FEATURE ENGINEERING] Cargando datos enriquecidos desde: {input_file}...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"No se encontró el archivo de entrada en: {input_file}")
        
    df = pd.read_csv(input_file)
    
    print("[FEATURE ENGINEERING] Creando variables de comportamiento temporal (Agregaciones)...")
    
    # Agregaciones vectorizadas basadas en el grupo de tarjeta 'card1'
    card1_mean_amt = df.groupby('card1')['TransactionAmt'].transform('mean')
    card1_std_amt = df.groupby('card1')['TransactionAmt'].transform('std').fillna(0)
    
    # Característica clave de detección: Relación de montos frente al histórico del usuario
    df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / (card1_mean_amt + 0.001)
    df['TransactionAmt_to_std_card1'] = df['TransactionAmt'] / (card1_std_amt + 0.001)
    
    # Conteo masivo de volumen transaccional (Inyectado en LightGBM con alta importancia)
    df['card1_transaction_count'] = df.groupby('card1')['TransactionID'].transform('count')
    
    print("[FEATURE ENGINEERING] Analizando consistencia de identidad y datos faltantes...")
    
    # Indicadores binarios de datos faltantes en variables críticas de huella digital
    df['missing_device_type'] = df['DeviceType'].isnull().astype(int)
    df['missing_device_info'] = df['DeviceInfo'].isnull().astype(int)
    
    # Tratamiento eficiente de variables categóricas de alta cardinalidad
    df['P_emaildomain'] = df['P_emaildomain'].fillna('unknown')
    
    # OPTIMIZACIÓN VECTORIZADA: Reemplazamos apply(lambda) por .isin() nativo de Pandas en C
    emails_comunes = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'anonymous.org', 'unknown']
    
    df['P_emaildomain_grouped'] = 'other' # Inicializar todo como 'other'
    mask_comunes = df['P_emaildomain'].isin(emails_comunes)
    df.loc[mask_comunes, 'P_emaildomain_grouped'] = df.loc[mask_comunes, 'P_emaildomain']
    
    # 2. Almacenamiento Seguro del Dataset Final
    print(f"[FEATURE ENGINEERING] Guardando dataset final con nuevas características en: {output_file}...")
    df.to_csv(output_file, index=False)
    print("[FEATURE ENGINEERING] PROCESO FINALIZADO CON ÉXITO.")

if __name__ == "__main__":
    print("==================================================================")
    print("== RUNNING FEATURE ENGINEERING PIPELINE: INJECTION VERSION ==")
    print("==================================================================")
    
    # Cargar los parámetros globales antes de iniciar las transformaciones
    config_global = cargar_configuracion()
    aplicar_feature_engineering_avanzado(config_global)