# Sistema Híbrido de Prevención de Fraude y Detección de Anomalías Transaccionales

Este proyecto implementa una solución de nivel de producción para la identificación temprana de fraude financiero utilizando un enfoque de **Inteligencia Artificial Híbrida (Neurosimbólica)**. El sistema está diseñado para proteger la salud financiera de la organización optimizando el balance entre la mitigación del riesgo monetario y la experiencia de usuario de los clientes legítimos.

---

## El Problema de Negocio

En el ecosistema financiero actual (Fintech, Neobancos y Procesadores de Pago), la prevención de fraude se enfrenta a un dilema operativo constante:

* **El Costo del Fraude:** Dejar pasar transacciones ilícitas genera pérdidas monetarias directas, multas operativas y contracargos (Chargebacks).
* **El Costo de la Fricción (Falsos Positivos):** Bloquear por error a un cliente honesto daña la reputación de la empresa, genera el abandono de la plataforma, satura los centros de soporte y provoca pérdidas por ventas legítimas no procesadas.

Los sistemas tradicionales basados únicamente en reglas fijas o modelos probabilísticos puros fallan debido a la capacidad de los defraudadores profesionales para imitar con alta precisión los hábitos de consumo y las estructuras de comportamiento de un usuario común.

---

## Análisis Exploratorio de Datos (EDA) e Impacto en el Negocio

Para fundamentar las decisiones de diseño de la arquitectura híbrida, se implementó un flujo de análisis exploratorio automatizado (`generar_eda.py`) y una bitácora de investigación interactiva (`eda_report.ipynb`). Los hallazgos estadísticos clave se describen a continuación:

### 1. Distribución y Desbalance Crítico de Clases
El fraude financiero representa únicamente un porcentaje aproximado al 3.5% del total de las operaciones transaccionales. Este desbalance estricto imposibilita el uso de modelos probabilísticos convencionales sin ajuste, ya que tenderían a ignorar la clase minoritaria para maximizar la precisión general.

* **Decisión de Ingeniería:** En lugar de emplear técnicas de sobremuestreo sintético (como SMOTE) que introducen ruido matemático en variables categóricas de alta cardinalidad, se optó por modificar la función de pérdida del optimizador de LightGBM mediante el parámetro `scale_pos_weight`. Esto penaliza con mayor rigor los errores sobre la clase fraudulenta sin alterar la distribución real del mercado.

### 2. Comportamiento del Monto Transaccional
El análisis de distribución intercuartílica demuestra que las medianas del monto financiero entre transacciones legítimas y fraudulentas son sumamente cercanas, concentrándose en rangos cotidianos de consumo. El fraude no se caracteriza simplemente por exhibir montos atípicamente elevados; los patrones de ataque operan mimetizándose dentro del comportamiento habitual del usuario para evadir reglas basadas en umbrales simples.

### 3. Aislamiento Espacial de Anomalías (Capa 1)
Al proyectar el monto de la transacción contra el volumen acumulado de operaciones por tarjeta (`card1_transaction_count`), el algoritmo no supervisado `Isolation Forest` logra trazar fronteras de aislamiento efectivas. Esta capa identifica y etiqueta con éxito ráfagas de transacciones consecutivas en ventanas de tiempo reducidas (ataques de velocidad o scripts automatizados), actuando como un filtro estadístico previo antes de la evaluación de variables de identidad complejas.

### 4. Justificación del Enfoque Híbrido Neurosimbólico
Dejar el control absoluto del flujo transaccional en un modelo de Machine Learning puro generaría un incremento crítico de falsos positivos en las zonas de alta incertidumbre estadística, bloqueando a usuarios legítimos y afectando la experiencia operativa. 

La integración de un **Motor de Reglas Lógicas Simbólicas** funciona como un validador de negocio de alta velocidad sobre los vectores de riesgo del modelo. Esta arquitectura coordinada mitiga el impacto operativo, asegurando la continuidad del negocio y la protección activa de los activos financieros de la organización.

---

## Arquitectura del Sistema en Capas

La solución procesa el flujo transaccional masivo a través de una canalización (pipeline) compuesta por **tres capas independientes y especializadas de seguridad**:

### Capa 1: El Radar Estadístico (Detección de Anomalías)
* **Función:** Evalúa la transacción de forma aislada y calcula la desviación matemática del movimiento basándose en la relación estructural entre montos y el identificador de la tarjeta (`card1`). Funciona en tiempo real sin requerir acceso inmediato al historial histórico profundo del usuario.
* **Algoritmo:** `Isolation Forest` (Enfoque No Supervisado).

### Capa 2: El Perfil de Comportamiento (Ingeniería de Características Temporal)
* **Función:** Analiza el contexto dinámico del usuario a través de variables agregadas de velocidad y volumen. Calcula ráfagas transaccionales (frecuencia de compras en ventanas de tiempo específicas) y desviaciones financieras respecto a los patrones habituales del cliente.
* **Tecnología:** Transformaciones relacionales optimizadas en `Pandas` y agregaciones vectoriales.

### Capa 3: El Intérprete Neurosimbólico (Fusión de Inteligencia Artificial y Reglas de Negocio)
* **Función:** Combina la potencia predictiva de un clasificador estadístico avanzado (que calcula el Score probabilístico de riesgo) con un motor inferencial de reglas lógicas duras operadas por expertos. Cuando el modelo probabilístico genera incertidumbre en sus fronteras de decisión (zona gris), el componente simbólico interviene para resolver el veredicto, minimizando los falsos positivos y discriminando con precisión el fraude real.
* **Algoritmo:** `LightGBM` (Light Gradient Boosting Machine) coordinado con lógica condicional jerárquica.

---

## Hallazgos e Impacto Real de Negocio

El sistema fue evaluado y validado utilizando el conjunto de datos masivo e histórico de **IEEE-CIS / Vesta Corporation** (compuesto por cerca de 600,000 registros transaccionales), simulando entornos de producción y transacciones futuras fuera del conjunto de entrenamiento.

La implementación del enfoque Neurosimbólico sobre los modelos tradicionales de Inteligencia Artificial arrojó los siguientes resultados operativos:

* **Protección Financiera Activa:** Intercepción automatizada de transacciones fraudulentas críticas desde el primer segundo de exposición en el pipeline de datos.
* **Reducción Masiva de Fricción Operativa:** Mientras que los modelos probabilísticos tradicionales bloqueaban erróneamente a **20,098** clientes legítimos debido a la dispersión de los datos, la arquitectura híbrida redujo esa cifra radicalmente a solo **7,575** casos sospechosos retenidos para validación interna.

> **Impacto Neto:** Se evitaron **12,523 bloqueos injustificados a usuarios honestos**, lo que mitiga directamente la tasa de abandono de clientes, protege el volumen de facturación neta de la organización y reduce drásticamente la carga de tickets de reclamación en las mesas de soporte operativo.

---

## Herramientas y Tecnologías Utilizadas

Para garantizar la escalabilidad, la velocidad de procesamiento en milisegundos y la mantenibilidad del software, se seleccionó un ecosistema tecnológico estándar en la industria:

* **Python 3.12:** Entorno de ejecución principal del sistema.
* **Pandas y NumPy:** Ingesta masiva de datos estructurados, optimización en el uso de memoria RAM mediante asignación correcta de tipos de datos, y cómputo de perfiles temporales vectorizados.
* **Scikit-Learn:** Framework utilizado para el modelado no supervisado de la Capa 1 y la partición robusta de datos.
* **LightGBM (Light Gradient Boosting Machine):** Algoritmo de ensamble basado en árboles de decisión optimizado por Microsoft, elegido por su alta eficiencia informática, bajo consumo de recursos de hardware y soporte nativo para datos altamente desbalanceados.
* **KaggleHub API:** Automatización del flujo de descarga, verificación e ingesta segura de las fuentes de datos crudas directamente desde los repositorios oficiales de la competencia IEEE-CIS.

---

## Estructura del Repositorio

El proyecto sigue las mejores prácticas de modularidad, separación de responsabilidades y arquitectura de código limpio en ingeniería de datos:

```text
├── data/
│   ├── raw/                  # Fuentes de datos originales y uniones de identidad en crudo.
│   └── processed/            # Datasets limpios enriquecidos con Scores de anomalias y variables agregadas.
├── images/                   # Graficos de alta resolucion generados por el flujo de produccion.
├── eda_report.ipynb          # Bitacora interactiva de investigacion, storytelling y analisis visual.
├── generar_eda.py            # Script ejecutable y automatizado para la generacion del reporte visual.
├── fraude.py / etl_pipeline.py  # Pipeline para el proceso de extraccion e ingesta inicial.
├── transform_avanzado.py     # Ingenieria de variables temporales y perfiles de velocidad transaccional.
├── entrenar_modelo.py        # Configuración, ajuste de hiperparametros y entrenamiento del clasificador probabilistico.
├── evaluar_capa1.py          # Scripts de validacion y calculo de metricas para el Isolation Forest.
├── sistema_neurosimbolico.py # Orquestador final que inyecta el motor de reglas logicas sobre el modelo probabilistico.
└── .gitignore                # Restricciones de exclusion para evitar el rastreo de archivos de datos masivos.