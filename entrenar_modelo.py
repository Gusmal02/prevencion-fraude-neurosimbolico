import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

DATA_PROCESSED_DIR = "data/processed"
INPUT_FILE = os.path.join(DATA_PROCESSED_DIR, "final_features_transactions.csv")

def entrenar_clasificador_fraude():
    print(f"[MODELADO] Cargando dataset de características finales desde: {INPUT_FILE}...")
    # Para la primera iteración rápida, seleccionamos las numéricas principales y las que creamos
    # Evitamos cargar las 430 columnas de golpe para validar la estabilidad del script
    columnas_clave = [
        'isFraud', 'TransactionDT', 'TransactionAmt', 'card1', 'card2',
        'is_anomaly', 'anomaly_score', 'TransactionAmt_to_mean_card1',
        'TransactionAmt_to_std_card1', 'card1_transaction_count',
        'missing_device_type', 'missing_device_info'
    ]
    
    df = pd.read_csv(INPUT_FILE, usecols=columnas_clave)
    
    print("[MODELADO] Preparando la división temporal de los datos (Train/Validation Split)...")
    # En fraude, ordenamos por tiempo (TransactionDT) y dividimos
    # Usaremos el 80% de los datos más antiguos para entrenar y el 20% más reciente para validar
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    X_train = train_df.drop(columns=['isFraud', 'TransactionDT'])
    y_train = train_df['isFraud']
    
    X_val = val_df.drop(columns=['isFraud', 'TransactionDT'])
    y_val = val_df['isFraud']
    
    print(f"[MODELADO] Registros de Entrenamiento: {X_train.shape[0]} | Fraudes: {y_train.sum()}")
    print(f"[MODELADO] Registros de Validación (Futuro): {X_val.shape[0]} | Fraudes: {y_val.sum()}")
    
    # Manejo de desbalance: Calculamos el peso de la clase para equilibrar el peso del fraude
    # lgb tiene el parámetro scale_pos_weight para esto
    ratio_no_fraude_a_fraude = (len(y_train) - y_train.sum()) / y_train.sum()
    
    print("[MODELADO] Configurando e Iniciando Entrenamiento con LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=ratio_no_fraude_a_fraude, # Ajuste estratégico por desbalance
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("[MODELADO] Entrenamiento finalizado con éxito.")
    
    # Predicciones
    print("\n[EVALUACIÓN DE MODELO] Generando métricas sobre los datos de validación...")
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]
    
    # Matriz de confusión
    print("\nMatriz de Confusión en Validación:")
    cm = confusion_matrix(y_val, preds)
    cm_df = pd.DataFrame(cm, index=['Real: Legítima', 'Real: Fraude'], columns=['Predicho: Legítima', 'Predicho: Fraude'])
    print(cm_df)
    
    # Reporte clásico
    print("\nReporte de Clasificación Metas de Negocio:")
    print(classification_report(y_val, preds, target_names=['Legítima', 'Fraude']))
    
    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_val, probs)
    area_bajo_curva = auc(recall, precision)
    print(f"Área Bajo la Curva Precision-Recall (AUPRC): {area_bajo_curva:.4f}")
    
    # Importancia de las variables (Para el factor de explicabilidad que querías ver)
    print("\n[EXPLICABILIDAD] Importancia de las Variables en las Decisiones:")
    importancias = pd.DataFrame({
        'Característica': X_train.columns,
        'Importancia': model.feature_importances_
    }).sort_values(by='Importancia', ascending=False)
    print(importancias)

if __name__ == "__main__":
    entrenar_clasificador_fraude()
    