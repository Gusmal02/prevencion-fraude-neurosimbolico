# Especificación Técnica de la Arquitectura: Pipeline ETL y Sistema Híbrido Neurosimbólico

Este documento detalla los componentes técnicos, decisiones de ingeniería de datos, fundamentos matemáticos y la evaluación del sistema de prevención de fraude a escala sobre el dataset **IEEE-CIS / Vesta Corporation**.

---

## 1. Arquitectura del Pipeline de Datos (ETL)

El sistema procesa volúmenes masivos de datos distribuidos en estructuras relacionales atomizadas. La canalización fue construida bajo un esquema modular estructurado en tres etapas definidas:

### Extract (Extracción Relacional)
* **Ingesta:** Automatizada mediante la API nativa de `kagglehub` en entornos aislados.
* **Fusión de Datos (Merge de Alta Cardinalidad):** Los datos crudos se dividen en características transaccionales (`train_transaction.csv`: 394 columnas) e identidades de red/dispositivo (`train_identity.csv`: 41 columnas).
* **Optimización de Memoria:** Se ejecuta una unión horizontal izquierda (*Left Join*) indexada estrictamente por la clave primaria `TransactionID`. Esto preserva la totalidad de los registros financieros ($590,540$), mapeando la huella digital correspondiente ($144,233$ registros) sin generar duplicaciones ni desbordamiento de memoria (*Out of Memory - OOM*) en el espacio de usuario.

$$\text{Dataset Final} = \text{Transaction} \bowtie_{\text{TransactionID}} \text{Identity}$$

### Transform (Procesamiento en Capas)
La fase de transformación ejecuta transformaciones deterministas de ingeniería de variables distribuidas en dos etapas secuenciales independientes para optimizar los tiempos de cómputo sobre la CPU:

#### Capa 1: Modelado No Supervisado (Aislamiento Estadístico)
Para inyectar propiedades predictivas globales sin sesgar el pipeline con la variable objetivo (`isFraud`), se ejecuta un algoritmo de **Isolation Forest** sobre un subconjunto de variables numéricas base:
* **Variables implicadas:** `TransactionAmt`, `TransactionDT`, `card1`, `card2`, `card3`, `card5`.
* **Manejo de nulos estratégicos:** Los campos con ausencia de datos se imputan con un valor bandera específico (`-999`). En este dominio, la falta de información denota anomalía estructural (ej. bloqueo de cookies o scripts automatizados).
* **Fundamento Matemático:** El algoritmo aísla observaciones construyendo árboles de decisión aleatorios. Las anomalías requieren menos particiones para ser aisladas en las ramas del árbol debido a su rareza estadística. El score de anomalía $s(x, n)$ se normaliza entre $0$ y $1$:

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

Donde $E(h(x))$ es la longitud promedio de la trayectoria de la observación $x$, y $c(n)$ es la longitud promedio de la trayectoria de un árbol fallido con $n$ nodos. Las observaciones con un `anomaly_score` altamente negativo o cercano a $1$ se indexan directamente mediante la nueva característica binaria `is_anomaly`.

#### Capa 2: Feature Engineering Avanzado (Perfiles Temporales)
Se simula el comportamiento de ventanas de tiempo continuas mediante agregaciones vectoriales calculadas mediante la función `.transform()` de Pandas para evitar bucles iterativos costosos:
* **Métricas de velocidad y volumen:** Agregación por el grupo emisor de tarjeta (`card1`) para obtener el volumen acumulado de transacciones concurrentes en el dataset (`card1_transaction_count`).
* **Desviación de Monto con respecto al Perfil:**

$$\text{TransactionAmt\_to\_mean\_card1} = \frac{\text{TransactionAmt}}{\mu_{\text{card1}} + 0.001}$$

$$\text{TransactionAmt\_to\_std\_card1} = \frac{\text{TransactionAmt}}{\sigma_{\text{card1}} + 0.001}$$

* **Consistencia de Canal e Identidad:** Tratamiento de variables categóricas de alta cardinalidad mediante agrupamiento por frecuencia (`P_emaildomain_grouped`) e indicadores lógicos de ausencia de datos de hardware (`missing_device_type`, `missing_device_info`).

### Load (Persistencia y Carga estructurada)
El DataFrame unificado, limpio y enriquecido con métricas de comportamiento temporal y aislamiento estadístico ($590,540 \text{ filas} \times 436 \text{ columnas}$ en su espectro completo, optimizado en su script de ejecución a variables de alto valor) se almacena localmente en `data/processed/final_features_transactions.csv` listo para el consumo de capas analíticas y modelos de inferencia.

---

## 2. Capa Predictiva Supervisada (LightGBM)

El núcleo predictivo del sistema se delegó a un clasificador basado en árboles de decisión optimizados por gradiente (**LightGBM - Light Gradient Boosting Machine**), seleccionado por su partición de hojas orientada por el gradiente (*Leaf-wise*) y su alto rendimiento computacional en matrices masivas.

### Estrategia de Validación Temporal
Para evitar el filtrado de información (*Data Leakage*) y replicar con fidelidad un entorno de producción, **no se utilizó una división aleatoria**. El dataset se ordenó estrictamente de forma cronológica por su marca de tiempo (`TransactionDT`), utilizando el primer $80\%$ de los registros históricos para entrenamiento y el $20\%$ posterior (transacciones del futuro) para validación.

* **Registros de Entrenamiento:** $472,432$ (Casos de Fraude: $16,599$)
* **Registros de Validación (Futuro):** $118,108$ (Casos de Fraude: $4,064$)

### Tratamiento del Desbalance de Clases
La tasa base de fraude en el set de entrenamiento es extremadamente asimétrica ($\approx 3.51\%$). Para contrarrestar la tendencia del optimizador a ignorar la clase minoritaria, se sintonizó el hiperparámetro matemático `scale_pos_weight`:

$$\text{scale\_pos\_weight} = \frac{N_{\text{negativos}}}{N_{\text{positivos}}} = \frac{455,833}{16,599} \approx 27.46$$

### Métricas del Clasificador Puro (LGBM)
Tras el entrenamiento, las predicciones probabilísticas directas sobre el conjunto del futuro arrojaron la siguiente matriz de confusión:

| | Predicho: Legítima | Predicho: Fraude |
|---|---|---|
| **Real: Legítima** | $93,946$ | $20,098$ (Falsos Positivos) |
| **Real: Fraude** | $1,804$ (Falsos Negativos) | $2,260$ (Verdaderos Positivos) |

* **Recall (Sensibilidad):** $0.56$ ($56\%$). Atrapa $2,260$ instancias de fraude real.
* **Precision (Precisión):** $0.10$ ($10\%$). Genera una tasa elevada de falsas alarmas ($20,098$ clientes bloqueados erróneamente).
* **AUPRC (Área Bajo la Curva Precision-Recall):** $0.1438$. Métrica de control clave para evaluar el rendimiento general sobre datos altamente desbalanceados.

---

## 3. Motor de Inferencia Neurosimbólico (Híbrido)

Para optimizar el costo operativo derivado de los $20,098$ falsos positivos producidos por el enfoque probabilístico puro de LightGBM, se implementó una **arquitectura neurosimbólica**. Esta capa combina la salida analítica continua del modelo (*Capa Neuronal/Gradiente*) con un sistema experto gobernado por reglas lógicas duras booleanas (*Capa Simbólica*).

El espectro de la probabilidad de salida $P(\text{Fraude} \mid x)$ se segmentó en tres zonas lógicas diferenciadas:

```text
[Probabilidad Inferencia LightGBM]
       │
       ├───> P < 0.15 ─────────────────────────> [ZONA VERDE]  ──> Autorizar Transmisión
       │
       ├───> P > 0.85 ─────────────────────────> [ZONA ROJA]   ──> Bloqueo Automatizado Inmediato
       │
       └───> 0.15 <= P <= 0.85 ────────────────> [ZONA GRIS]   ──> Evaluar Reglas Simbólicas Expertas
                                                      │
                                                      ├──> ¿(Monto Anormal AND ID Oculta) OR Anomalía Aislada?
                                                      │           ├──> SÍ ──> Bloquear Transacción
                                                      │           └──> NO ──> Permitir Comprar / Alerta OTP
