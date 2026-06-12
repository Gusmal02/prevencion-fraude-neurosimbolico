import os
import pandas as pd
import numpy as np

DATA_PROCESSED_DIR = "data/processed"
INPUT_FILE = os.path.join(DATA_PROCESSED_DIR, "enriched_transactions.csv")
OUTPUT_FILE = os.path.join(DATA_PROCESSED_DIR, "final_features_transactions.csv")

def aplicar_feature_engineering_avanzado():
    print(f"[FEATURE ENGINEERING] Cargando datos enriquecidos desde: {INPUT_FILE}...")
    # Cargamos el dataset generado en el paso anterior (que ya incluye el score de anomalía)
    df = pd.read_csv(INPUT_FILE)
    
    print("[FEATURE ENGINEERING] Creando variables de comportamiento temporal (Agregaciones)...")
    
    # 1. Agregaciones basadas en el grupo de tarjeta 'card1'
    # Calculamos el promedio del monto de transacción por tarjeta
    card1_mean_amt = df.groupby('card1')['TransactionAmt'].transform('mean')
    # Calculamos la desviación estándar del monto por tarjeta (llenamos nulos con 0)
    card1_std_amt = df.groupby('card1')['TransactionAmt'].transform('std').fillna(0)
    
    # Creamos la característica: relación entre el monto actual y el promedio de la tarjeta
    # Si es mucho mayor a 1, significa un gasto inusualmente alto para ese usuario
    df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / (card1_mean_amt + 0.001)
    df['TransactionAmt_to_std_card1'] = df['TransactionAmt'] / (card1_std_amt + 0.001)
    
    # Conteo de transacciones por tarjeta en todo el dataset para medir volumen
    df['card1_transaction_count'] = df.groupby('card1')['TransactionID'].transform('count')
    
    print("[FEATURE ENGINEERING] Analizando consistencia de identidad y datos faltantes...")
    
    # 2. Indicadores binarios de datos faltantes en variables críticas de huella digital
    # Si falta DeviceType o DeviceInfo, guardamos un 1 (potencialmente sospechoso)
    df['missing_device_type'] = df['DeviceType'].isnull().astype(int)
    df['missing_device_info'] = df['DeviceInfo'].isnull().astype(int)
    
    # 3. Tratamiento de variables categóricas de alta cardinalidad (ej. Proveedor de correo electrónico)
    # Rellenamos nulos con 'unknown' antes de procesar
    df['P_emaildomain'] = df['P_emaildomain'].fillna('unknown')
    
    # Agrupamos correos comunes y marcamos los de infraestructura sospechosa o inusual
    # Esto reduce la cardinalidad para no saturar el modelo final con One-Hot Encoding masivo
    emails_comunes = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'anonymous.org', 'unknown']
    df['P_emaildomain_grouped'] = df['P_emaildomain'].apply(lambda x: x if x in emails_comunes else 'other')
    
    print("[FEATURE ENGINEERING] Seleccionando y limpiando variables finales para el modelo...")
    
    # Almacenamos el dataset con las nuevas columnas estratégicas
    # En este punto conservamos 'isFraud' como nuestra variable objetivo (target)
    print(f"[FEATURE ENGINEERING] Guardando dataset final con nuevas características en: {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("[FEATURE ENGINEERING] PROCESO FINALIZADO CON ÉXITO.")

if __name__ == "__main__":
    aplicar_feature_engineering_avanzado()