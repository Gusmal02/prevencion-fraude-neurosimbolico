import os
import sys
import logging
import yaml
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Configuración del Logger 
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

def configurar_entorno():
    """Establece los estilos visuales globales de las gráficas."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

def extraer_datos_cache(config):
    """Localiza y extrae los datos manejando excepciones de red y estructura de archivos."""
    logging.info("[ETL - EXTRACT] Conectando con la API de Kagglehub para localizar el dataset...")
    try:
        # Descarga protegida contra fallas de API/Red
        download_path = kagglehub.dataset_download("lnasiri007/ieeecis-fraud-detection")
        logging.info(f"[ETL - EXTRACT] Repositorio verificado en caché: {download_path}")
        files_in_download = os.listdir(download_path)
    except Exception as e:
        logging.critical(f"Falla crítica al interactuar con Kaggle Hub API: {e}")
        sys.exit(1)
    
    # Identificación dinámica de archivos
    transaction_file = next((f for f in files_in_download if 'transaction' in f.lower() and 'train' in f.lower()), None)
    identity_file = next((f for f in files_in_download if 'identity' in f.lower() and 'train' in f.lower()), None)
    
    if not transaction_file or not identity_file:
        logging.error("No se encontraron los archivos CSV requeridos de entrenamiento o identidad.")
        sys.exit(1)
        
    tx_path = os.path.join(download_path, transaction_file)
    id_path = os.path.join(download_path, identity_file)
    
    logging.info("[ETL - EXTRACT] Cargando datasets optimizados en memoria (100k filas)...")
    try:
        tx = pd.read_csv(tx_path, nrows=100000)
        id_df = pd.read_csv(id_path, nrows=100000)
    except Exception as e:
        logging.error(f"Error al leer los archivos CSV en disco: {e}")
        sys.exit(1)
    
    logging.info("[ETL - TRANSFORM] Ejecutando fusión relacional por TransactionID...")
    df = pd.merge(tx, id_df, on='TransactionID', how='left')
    logging.info(f"[ETL - TRANSFORM] Datos listos para graficación. Dimensiones: {df.shape[0]} filas.")
    return df

# ---- FUNCIONES MODULARIZADAS DE GRAFICACIÓN  ----

def graficar_desbalance_clases(df, output_dir='images'):
    """Genera y guarda el gráfico de distribución de clases para análisis de desbalance."""
    logging.info("[EDA - BI] Generando gráfico de distribución de clases...")
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='isFraud', data=df, hue='isFraud', palette='Set2', legend=False)
    plt.title('Distribución Crítica de Clases\n(Desbalance Financiero Estricto)', fontsize=12, fontweight='bold')
    plt.xlabel('¿Es Fraude? (0 = Legítima, 1 = Fraude)')
    plt.ylabel('Cantidad de Transacciones')
    plt.xticks([0, 1], ['Legítimas', 'Fraude'])

    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.2f}%'
        x_coord = p.get_x() + p.get_width() / 2 - 0.1
        y_coord = p.get_height() + (total * 0.01)
        ax.annotate(percentage, (x_coord, y_coord), fontsize=10, fontweight='bold')
    
    plt.savefig(os.path.join(output_dir, '01_distribucion_clases.png'), dpi=300, bbox_inches='tight')
    plt.close()

def graficar_densidad_montos(df, output_dir='images'):
    """Genera y guarda el boxplot analítico de montos por categoría de riesgo."""
    logging.info("[EDA - BI] Generando gráfico de densidad de montos transaccionales...")
    plt.figure(figsize=(8, 5))
    df_filtrado = df[df['TransactionAmt'] < 500]
    sns.boxplot(x='isFraud', y='TransactionAmt', data=df_filtrado, hue='isFraud', palette='Pastel1', legend=False)
    plt.title('Análisis de la Densidad del Monto por Categoría de Riesgo', fontsize=12, fontweight='bold')
    plt.xlabel('Clasificación de la Transacción')
    plt.ylabel('Monto de la Operación ($ USD)')
    plt.xticks([0, 1], ['Legítimas', 'Fraude'])
    
    plt.savefig(os.path.join(output_dir, '02_analisis_densidad_montos.png'), dpi=300, bbox_inches='tight')
    plt.close()

def graficar_radar_anomalias(df, config, output_dir='images'):
    """Entrena y grafica la Capa 1 usando hiperparámetros dinámicos desde config.yaml."""
    logging.info("[EDA - BI] Procesando muestreo y entrenamiento del Isolation Forest para análisis espacial...")
    
    # Ingeniería de variables temporal
    df['card1_transaction_count'] = df.groupby('card1')['TransactionID'].transform('count')
    df_sample = df.dropna(subset=['TransactionAmt', 'card1_transaction_count']).sample(20000, random_state=42)

    # Inyección dinámica de hiperparámetros desde la configuración centralizada
    iso_config = config['isolation_forest']
    iso = IsolationForest(
        contamination=iso_config['contamination'], 
        n_estimators=iso_config['n_estimators'],
        random_state=iso_config['random_state']
    )
    
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
    plt.title('Capa 1: Aislamiento Estadístico del Espacio Transaccional', fontsize=12, fontweight='bold')
    plt.xlabel('Monto de la Transacción ($ USD)')
    plt.ylabel('Volumen Acumulado de la Tarjeta (card1_count)')
    plt.legend(title='Veredicto del Radar', labels=['Estructura Normal', 'Anomalía Estructural Detectada'])
    
    plt.savefig(os.path.join(output_dir, '03_radar_anomalias_espacial.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 1. Carga de configuraciones Centralizadas
    config = cargar_configuracion()
    
    # 2. Inicialización del entorno visual e imágenes
    configurar_entorno()
    os.makedirs('images', exist_ok=True)
    
    # 3. Pipelines de Ingesta protegida
    dataset = extraer_datos_cache(config)
    
    # 4. Procesamiento modular de los entregables visuales
    graficar_desbalance_clases(dataset)
    graficar_densidad_montos(dataset)
    graficar_radar_anomalias(dataset, config)
    
    logging.info("[EDA - END] Proceso completado con éxito. Entregables listos en 'images/'.")

if __name__ == "__main__":
    main()