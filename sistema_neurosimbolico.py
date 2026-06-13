import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, classification_report
# Respetamos el principio de Inyección de Dependencias
from generar_eda import cargar_configuracion

def ejecutar_sistema_neurosimbolico(config):
    print("\n[NEUROSYMBOLIC] Cargando configuración e inicializando el sistema híbrido...")
    
    # 1. Recuperar rutas y parámetros dinámicamente desde el config.yaml real
    processed_dir = config['data']['processed_dir']
    input_file = os.path.join(processed_dir, "final_features_transactions.csv")
    
    # Extraemos parámetros de hiperparametrización y del motor de reglas en inglés
    lgb_params = config['lightgbm']
    rules_params = config['rules']
    
    columnas_clave = [
        'isFraud', 'TransactionDT', 'TransactionAmt', 'card1', 'card2',
        'is_anomaly', 'anomaly_score', 'TransactionAmt_to_mean_card1',
        'TransactionAmt_to_std_card1', 'card1_transaction_count',
        'missing_device_type', 'missing_device_info'
    ]
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"No se encontró el dataset final en: {input_file}")
        
    df = pd.read_csv(input_file, usecols=columnas_clave).sort_values('TransactionDT').reset_index(drop=True)
    
    # Split Temporal (80% Train / 20% Validation)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:].copy()
    
    X_train = train_df.drop(columns=['isFraud', 'TransactionDT'])
    y_train = train_df['isFraud']
    X_val = val_df.drop(columns=['isFraud', 'TransactionDT'])
    y_val = val_df['isFraud']
    
    # Balanceo dinámico basado en los datos de entrenamiento actuales
    ratio = (len(y_train) - y_train.sum()) / y_train.sum()
    
    print("[NEUROSYMBOLIC] Re-entrenando componente probabilístico base (LightGBM)...")
    model = lgb.LGBMClassifier(
        n_estimators=lgb_params['n_estimators'],
        learning_rate=lgb_params['learning_rate'],
        num_leaves=lgb_params['num_leaves'],
        scale_pos_weight=ratio,
        random_state=lgb_params['random_state'],
        verbose=-1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Extraemos las probabilidades puras del clasificador
    val_df['prob_fraude'] = model.predict_proba(X_val)[:, 1]
    
    print("[NEUROSYMBOLIC] Aplicando Motor de Reglas Simbólicas vectorizado sobre la Zona Gris...")
    
    # 2. EVALUACIÓN VECTORIZADA (Optimización de rendimiento para producción)
    # Inicializamos todas las decisiones por defecto en 0 (Permitir)
    val_df['decision_final'] = 0
    
    # Extraemos los umbrales lógicos inyectados desde el YAML
    p_low = rules_params['prob_low_threshold']      # Por ejemplo: 0.15
    p_high = rules_params['prob_high_threshold']    # Por ejemplo: 0.85
    monto_ratio_max = rules_params['high_risk_amount_ratio'] # Por ejemplo: 3.0
    score_isof_critico = rules_params['critical_anomaly_score'] # Por ejemplo: -0.65
    
    # CAPA 1: Regla Probabilística de Extremos Duros
    val_df.loc[val_df['prob_fraude'] > p_high, 'decision_final'] = 1  # Fraude Directo Seguro
    
    # CAPA 2: Lógica Simbólica Avanzada aplicada estrictamente sobre la Zona Gris
    zona_gris_mask = (val_df['prob_fraude'] >= p_low) & (val_df['prob_fraude'] <= p_high)
    
    condicion_dispositivo_oculto = (val_df['missing_device_type'] == 1) | (val_df['missing_device_info'] == 1)
    condicion_monto_anormal = val_df['TransactionAmt_to_mean_card1'] > monto_ratio_max
    condicion_aislamiento_critico = val_df['anomaly_score'] < score_isof_critico
    
    # Combinación simbólica booleana
    regla_fraude_zona_gris = (condicion_monto_anormal & condicion_dispositivo_oculto) | condicion_aislamiento_critico
    
    # Aplicamos el bloqueo a las transacciones de la zona gris que cumplan las reglas duras
    val_df.loc[zona_gris_mask & regla_fraude_zona_gris, 'decision_final'] = 1
    
    # 3. Evaluación del Impacto de Negocio del Sistema Híbrido
    print("\n================ METRICAS DEL SISTEMA NEUROSIMBÓLICO ================")
    cm_hybrid = confusion_matrix(y_val, val_df['decision_final'])
    cm_df = pd.DataFrame(cm_hybrid, index=['Real: Legítima', 'Real: Fraude'], columns=['Sistema: Permitir', 'Sistema: Bloquear'])
    print(cm_df)
    
    print("\nReporte de Clasificación Final:")
    print(classification_report(y_val, val_df['decision_final'], target_names=['Legítima', 'Fraude']))
    
    # Comparativa basada en la matriz previa de falsos positivos
    falsos_positivos_anteriores = 20256
    falsos_positivos_actuales = cm_hybrid[0, 1]
    reduccion = falsos_positivos_anteriores - falsos_positivos_actuales
    
    print(f"=== IMPACTO EN LA EXPERIENCIA DEL USUARIO ===")
    print(f"Falsos Positivos anteriores (LightGBM puro): {falsos_positivos_anteriores}")
    print(f"Falsos Positivos actuales (Híbrido Neurosimbólico): {falsos_positivos_actuales}")
    print(f"Reducción de fricción: ¡Se evitaron {reduccion} bloqueos erróneos a clientes legítimos!")
    print("=====================================================================")

if __name__ == "__main__":
    print("==================================================================")
    print("== RUNNING HYBRID NEURO-SYMBOLIC ENGINE: INTERPRETER VERSION ==")
    print("==================================================================")
    
    # Cargar los parámetros globales antes de la orquestación
    config_global = cargar_configuracion()
    ejecutar_sistema_neurosimbolico(config_global)