import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, classification_report

DATA_PROCESSED_DIR = "data/processed"
INPUT_FILE = os.path.join(DATA_PROCESSED_DIR, "final_features_transactions.csv")

def ejecutar_sistema_neurosimbolico():
    print("[NEUROSYMBOLIC] Cargando características finales y entrenando base...")
    
    # 1. Cargamos y preparamos las variables (mismo split temporal)
    columnas_clave = [
        'isFraud', 'TransactionDT', 'TransactionAmt', 'card1', 'card2',
        'is_anomaly', 'anomaly_score', 'TransactionAmt_to_mean_card1',
        'TransactionAmt_to_std_card1', 'card1_transaction_count',
        'missing_device_type', 'missing_device_info'
    ]
    df = pd.read_csv(INPUT_FILE, usecols=columnas_clave).sort_values('TransactionDT').reset_index(drop=True)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:].copy() # Trabajamos sobre la copia de validación
    
    X_train = train_df.drop(columns=['isFraud', 'TransactionDT'])
    y_train = train_df['isFraud']
    X_val = val_df.drop(columns=['isFraud', 'TransactionDT'])
    y_val = val_df['isFraud']
    
    # Re-entrenamos rápidamente el LightGBM para obtener las probabilidades base
    ratio = (len(y_train) - y_train.sum()) / y_train.sum()
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, scale_pos_weight=ratio, random_state=42, verbose=-1, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Extraemos las probabilidades puras del componente neuronal/estadístico
    val_df['prob_fraude'] = model.predict_proba(X_val)[:, 1]
    
    print("\n[NEUROSYMBOLIC] Aplicando Motor de Reglas Simbólicas sobre la Zona Gris...")
    
    # 2. Definición del Motor Simbólico (Reglas lógicas del experto)
    veredictos_finales = []
    
    for idx, row in val_df.iterrows():
        p = row['prob_fraude']
        
        # CAPA 1: Regla Estadística de Extremos
        if p < 0.15:
            veredicto = 0 # Legítima segura
        elif p > 0.85:
            veredicto = 1 # Fraude seguro (Bloqueo inmediato)
            
        # CAPA 2: Evaluación Simbólica en la Zona Gris (0.15 <= p <= 0.85)
        else:
            # Regla lógica A: Si el monto supera por mucho la media del usuario Y ocultó su dispositivo
            condicion_dispositivo_oculto = (row['missing_device_type'] == 1) or (row['missing_device_info'] == 1)
            condicion_monto_anormal = row['TransactionAmt_to_mean_card1'] > 3.0
            
            # Regla lógica B: Si el Isolation Forest lo considera una anomalía severa (score muy bajo)
            condicion_aislamiento_critico = row['anomaly_score'] < -0.65
            
            # Inferencia lógica combinada (Símbolos booleanos)
            if (condicion_monto_anormal and condicion_dispositivo_oculto) or condicion_aislamiento_critico:
                veredicto = 1 # Activación de regla dura: Bloquear
            else:
                veredicto = 0 # No cumple las condiciones de peligro extremo: Permitir transaccionar
                
        veredictos_finales.append(veredicto)
        
    val_df['decision_final'] = veredictos_finales
    
    # 3. Evaluación del Impacto de Negocio del Sistema Híbrido
    print("\n================ METRICAS DEL SISTEMA NEUROSIMBÓLICO ================")
    cm_hybrid = confusion_matrix(y_val, val_df['decision_final'])
    cm_df = pd.DataFrame(cm_hybrid, index=['Real: Legítima', 'Real: Fraude'], columns=['Sistema: Permitir', 'Sistema: Bloquear'])
    print(cm_df)
    
    print("\nReporte de Clasificación Final:")
    print(classification_report(y_val, val_df['decision_final'], target_names=['Legítima', 'Fraude']))
    
    # Comparativa de Falsos Positivos contra el modelo anterior
    falsos_positivos_anteriores = 20098
    falsos_positivos_actuales = cm_hybrid[0, 1]
    reduccion = falsos_positivos_anteriores - falsos_positivos_actuales
    
    print(f"=== IMPACTO EN LA EXPERIENCIA DEL USUARIO ===")
    print(f"Falsos Positivos anteriores (LightGBM puro): {falsos_positivos_anteriores}")
    print(f"Falsos Positivos actuales (Híbrido Neurosimbólico): {falsos_positivos_actuales}")
    print(f"Reducción de fricción: ¡Se evitaron {reduccion} bloqueos erróneos a clientes legítimos!")
    print("=====================================================================")

if __name__ == "__main__":
    ejecutar_sistema_neurosimbolico()