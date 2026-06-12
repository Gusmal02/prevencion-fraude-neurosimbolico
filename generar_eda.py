import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

def configurar_entorno():
    """Establece los estilos visuales globales de las graficas."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

def extraer_datos_cache():
    """Conecta con la API de Kagglehub para localizar y cargar los datos con limites de optimizacion."""
    print("[ETL - EXTRACT] Localizando repositorio en cache global...")
    download_path = kagglehub.dataset_download("lnasiri007/ieeecis-fraud-detection")
    files_in_download = os.listdir(download_path)
    
    transaction_file = next((f for f in files_in_download if 'transaction' in f.lower() and 'train' in f.lower()), None)
    identity_file = next((f for f in files_in_download if 'identity' in f.lower() and 'train' in f.lower()), None)
    
    tx_path = os.path.join(download_path, transaction_file)
    id_path = os.path.join(download_path, identity_file)
    
    # Optimizacion radical: Leemos una fraccion masiva pero controlada (100,000 filas) 
    # para no saturar el buffer ni la memoria RAM del sistema
    print("[ETL - EXTRACT] Cargando datasets optimizados en memoria (100k filas)...")
    tx = pd.read_csv(tx_path, nrows=100000)
    id_df = pd.read_csv(id_path, nrows=100000)
    
    print("[ETL - TRANSFORM] Ejecutando fusion relacional por TransactionID...")
    df = pd.merge(tx, id_df, on='TransactionID', how='left')
    print(f"[ETL - TRANSFORM] Datos listos para graficacion. Dimensiones: {df.shape[0]} filas.")
    return df

def procesar_y_guardar_graficas(df):
    """Genera las visualizaciones analiticas y las persiste en formato PNG."""
    # Asegurar la existencia de la carpeta para las imagenes del repositorio
    os.makedirs('images', exist_ok=True)
    print("[EDA - BI] Directorio 'images/' verificado.")

    # ---- 1. GRAFICA: Desbalance de Clases ----
    print("[EDA - BI] Generando grafico de distribucion de clases...")
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='isFraud', data=df, hue='isFraud', palette='Set2', legend=False)
    plt.title('Distribucion Critica de Clases\n(Desbalance Financiero Estricto)', fontsize=12, fontweight='bold')
    plt.xlabel('¿Es Fraude? (0 = Legitima, 1 = Fraude)')
    plt.ylabel('Cantidad de Transacciones')
    plt.xticks([0, 1], ['Legitimas', 'Fraude'])

    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.2f}%'
        x_coord = p.get_x() + p.get_width() / 2 - 0.1
        y_coord = p.get_height() + 5000
        ax.annotate(percentage, (x_coord, y_coord), fontsize=10, fontweight='bold')
    
    plt.savefig('images/01_distribucion_clases.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---- 2. GRAFICA: Boxplot de Montos ----
    print("[EDA - BI] Generando grafico de densidad de montos transaccionales...")
    plt.figure(figsize=(8, 5))
    df_filtrado = df[df['TransactionAmt'] < 500]
    sns.boxplot(x='isFraud', y='TransactionAmt', data=df_filtrado, hue='isFraud', palette='Pastel1', legend=False)
    plt.title('Analis de la Densidad del Monto por Categoria de Riesgo', fontsize=12, fontweight='bold')
    plt.xlabel('Clasificacion de la Transaccion')
    plt.ylabel('Monto de la Operacion ($ USD)')
    plt.xticks([0, 1], ['Legitimas', 'Fraude'])
    
    plt.savefig('images/02_analisis_densidad_montos.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---- 3. GRAFICA: Validacion del Radar de Anomalias ----
    print("[EDA - BI] Procesando muestreo y entrenamiento del Isolation Forest para analisis espacial...")
    df['card1_transaction_count'] = df.groupby('card1')['TransactionID'].transform('count')
    df_sample = df.dropna(subset=['TransactionAmt', 'card1_transaction_count']).sample(20000, random_state=42)

    iso = IsolationForest(contamination=0.04, random_state=42)
    preds = iso.fit_predict(df_sample[['TransactionAmt', 'card1_transaction_count']])
    df_sample['is_anomaly'] = np.where(preds == -1, 1, 0)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='TransactionAmt', 
        y='card1_transaction_count', 
        hue='is_anomaly', 
        data=df_sample[df_sample['TransactionAmt'] < 1000],
        palette={0: '#2ecc71', 1: '#e74c3c'}, 
        alpha=0.6,
        edgecolor='none'
    )
    plt.title('Capa 1: Aislamiento Estadistico del Espacio Transaccional', fontsize=12, fontweight='bold')
    plt.xlabel('Monto de la Transaccion ($ USD)')
    plt.ylabel('Volumen Acumulado de la Tarjeta (card1_count)')
    plt.legend(title='Veredicto del Radar', labels=['Estructura Normal', 'Anomalia Estructural Detectada'])
    
    plt.savefig('images/03_radar_anomalias_espacial.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[EDA - END] Proceso completado. Las tres imagenes fueron exportadas a la carpeta 'images/'.")

if __name__ == "__main__":
    configurar_entorno()
    dataset = extraer_datos_cache()
    procesar_y_guardar_graficas(dataset)