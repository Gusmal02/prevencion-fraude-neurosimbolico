import os
import sys
import logging
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

# Configuración del Logger Profesional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def cargar_configuracion(ruta_config="config.yaml"):
    """Carga de forma segura el archivo de configuración centralizado."""
    if not os.path.exists(ruta_config):
        logging.error(f"Archivo de configuración no encontrado en: {ruta_config}")
        sys.exit(1)
    try:
        with open(ruta_config, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error al parsear el archivo YAML: {e}")
        sys.exit(1)

def entrenar_clasificador_fraude():
    # 1. Cargar configuración e inicializar rutas
    config = cargar_configuracion()
    
    # Asignación dinámica desde config.yaml o fallback si usas el archivo extendido
    input_file = os.path.join(config['data']['processed_dir'], "final_features_transactions.csv")
    
    # Blindaje de rutas
    if not os.path.exists(input_file):
        logging.critical(f"Archivo de características procesadas no encontrado en: {input_file}")
        logging.info("Asegúrate de ejecutar el pipeline de procesamiento antes del entrenamiento.")
        sys.exit(1)
        
    logging.info(f"[MODELADO] Cargando dataset de características finales desde: {input_file}...")
    
    columnas_clave = [
        'isFraud', 'TransactionDT', 'TransactionAmt', 'card1', 'card2',
        'is_anomaly', 'anomaly_score', 'TransactionAmt_to_mean_card1',
        'TransactionAmt_to_std_card1', 'card1_transaction_count',
        'missing_device_type', 'missing_device_info'
    ]
    
    try:
        df = pd.read_csv(input_file, usecols=columnas_clave)
    except Exception as e:
        logging.error(f"Error al leer las columnas clave del archivo: {e}")
        sys.exit(1)
    
    logging.info("[MODELADO] Preparando la división temporal de los datos (Train/Validation Split)...")
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    X_train = train_df.drop(columns=['isFraud', 'TransactionDT'])
    y_train = train_df['isFraud']
    
    X_val = val_df.drop(columns=['isFraud', 'TransactionDT'])
    y_val = val_df['isFraud']
    
    logging.info(f"[MODELADO] Registros Entrenamiento: {X_train.shape[0]} | Fraudes: {y_train.sum()}")
    logging.info(f"[MODELADO] Registros Validación (Futuro): {X_val.shape[0]} | Fraudes: {y_val.sum()}")
    
    # Cálculo dinámico del desbalance
    ratio_no_fraude_a_fraude = (len(y_train) - y_train.sum()) / y_train.sum()
    
    # ---- 🚀 INICIO DEL RASTREO CON MLFLOW ----
    logging.info("[MLFLOW] Inicializando experimento de Gobernanza de Modelos...")
    mlflow.set_experiment("Prevencion_Fraude_Neurosimbolico")
    
    with mlflow.start_run() as run:
        # Obtener hiperparámetros desde config.yaml
        lgb_config = config['lightgbm']
        
        logging.info("[MODELADO] Configurando e Iniciando Entrenamiento con LightGBM...")
        model = lgb.LGBMClassifier(
            n_estimators=lgb_config['n_estimators'],
            learning_rate=lgb_config['learning_rate'],
            num_leaves=lgb_config['num_leaves'],
            scale_pos_weight=ratio_no_fraude_a_fraude, # Ajuste estratégico dinámico
            random_state=lgb_config['random_state'],
            n_jobs=-1
        )
        
        # MLflow registra automáticamente parámetros de LightGBM
        mlflow.log_param("scale_pos_weight_calculado", ratio_no_fraude_a_fraude)
        mlflow.log_param("n_estimators", lgb_config['n_estimators'])
        mlflow.log_param("learning_rate", lgb_config['learning_rate'])
        mlflow.log_param("num_leaves", lgb_config['num_leaves'])
        
        model.fit(X_train, y_train)
        logging.info("[MODELADO] Entrenamiento finalizado con éxito.")
        
        # Inferencia en Validación
        logging.info("[EVALUACIÓN DE MODELO] Generando métricas sobre los datos de validación...")
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        
        # Matriz de confusión
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()
        
        # Registrar métricas numéricas exactas en la bitácora de MLflow
        mlflow.log_metric("True_Negatives", tn)
        mlflow.log_metric("False_Positives", fp)
        mlflow.log_metric("False_Negatives", fn)
        mlflow.log_metric("True_Positives", tp)
        
        # Curva Precision-Recall y AUPRC
        precision, recall, _ = precision_recall_curve(y_val, probs)
        area_bajo_curva = auc(recall, precision)
        mlflow.log_metric("AUPRC", area_bajo_curva)
        
        # Mostrar en consola para control local
        print("\nMatriz de Confusión en Validación:")
        cm_df = pd.DataFrame(cm, index=['Real: Legítima', 'Real: Fraude'], columns=['Predicho: Legítima', 'Predicho: Fraude'])
        print(cm_df)
        
        print("\nReporte de Clasificación Metas de Negocio:")
        reporte_str = classification_report(y_val, preds, target_names=['Legítima', 'Fraude'])
        print(reporte_str)
        print(f"Área Bajo la Curva Precision-Recall (AUPRC): {area_bajo_curva:.4f}")
        
        # Guardar reporte de texto como artefacto en MLflow
        with open("classification_report.txt", "w") as f:
            f.write(reporte_str)
        mlflow.log_artifact("classification_report.txt")
        os.remove("classification_report.txt")
        
        # Guardado y registro del binario del modelo en MLflow
        logging.info("[MLFLOW] Serializando y registrando artefacto final del modelo en la nube local...")
        mlflow.lightgbm.log_model(model, artifact_path="model")
        
        # Importancia de las variables (Para el factor de explicabilidad)
        importancias = pd.DataFrame({
            'Característica': X_train.columns,
            'Importancia': model.feature_importances_
        }).sort_values(by='Importancia', ascending=False)
        print("\n[EXPLICABILIDAD] Importancia de las Variables en las Decisiones:")
        print(importancias)

if __name__ == "__main__":
    entrenar_clasificador_fraude()