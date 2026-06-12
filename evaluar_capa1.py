import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

DATA_PROCESSED_DIR = "data/processed"
FILE_PATH = os.path.join(DATA_PROCESSED_DIR, "enriched_transactions.csv")

def evaluar_deteccion_anomalias():
    print(f"[EVALUACIÓN] Cargando datos enriquecidos desde: {FILE_PATH}...")
    # Leemos únicamente las columnas necesarias para no saturar memoria
    df = pd.read_csv(FILE_PATH, usecols=['isFraud', 'is_anomaly', 'anomaly_score'])
    
    print("\n[EVALUACIÓN] Matriz de Confusión (isFraud vs is_anomaly):")
    # isFraud es el suelo real (Ground Truth), is_anomaly es nuestra predicción ciega
    cm = confusion_matrix(df['isFraud'], df['is_anomaly'])
    
    # Estructuramos la matriz de forma visual para el portafolio
    cm_df = pd.DataFrame(cm, 
                         index=['Real: No Fraude (Legítima)', 'Real: Fraude'], 
                         columns=['Predicho: Normal', 'Predicho: Anómalo'])
    print(cm_df)
    
    print("\n[EVALUACIÓN] Reporte de Clasificación detallado:")
    print(classification_report(df['isFraud'], df['is_anomaly'], target_names=['Legítima', 'Fraude']))
    
    # Calcular cuántos fraudes reales atrapamos del total existente (Recall)
    total_fraudes_reales = df['isFraud'].sum()
    fraudes_atrapados = cm[1, 1]
    porcentaje_atrapado = (fraudes_atrapados / total_fraudes_reales) * 100
    
    print(f"\n=== CONCLUSIÓN DE NEGOCIO ===")
    print(f"El sistema detectó {total_fraudes_reales} fraudes reales en total.")
    print(f"El Isolation Forest, actuando COMPLETAMENTE A CIEGAS (sin conocer las etiquetas),")
    print(f"logró interceptar {fraudes_atrapados} transacciones fraudulentas ({porcentaje_atrapado:.2f}% del total).")

if __name__ == "__main__":
    evaluar_deteccion_anomalias()