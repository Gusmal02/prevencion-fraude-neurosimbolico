# Sistema Híbrido de Prevención de Fraude y Detección de Anomalías Transaccionales

Este proyecto implementa una solución de nivel producción para la identificación temprana de fraude financiero utilizando un enfoque de **Inteligencia Artificial Híbrida (Neurosimbólica)**. El sistema está diseñado para proteger la salud financiera de la organización sin arruinar la experiencia de compra de los usuarios legítimos.

---

## 🎯 El Problema de Negocio

En el ecosistema financiero actual (Fintech / Tarjetas de Crédito), la prevención de fraude se enfrenta a un dilema constante:

* **El costo del fraude:** Dejar pasar transacciones ilícitas genera pérdidas monetarias directas y contracargos.
* **El costo de la fricción:** Bloquear por error a un cliente honesto (Falso Positivo) daña la reputación de la empresa, satura el centro de soporte y genera pérdidas por ventas legítimas no procesadas.

Los sistemas tradicionales basados únicamente en estadísticas fallan porque los defraudadores profesionales aprenden a imitar con gran precisión los hábitos de un usuario común.

---

## 🛠️ Nuestra Solución: Arquitectura en Capas

Para solucionar esto, diseñamos un pipeline (canalización de datos) inteligente que procesa las transacciones a través de **tres capas independientes de seguridad**:

### Capa 1: El Radar de Sospecha (Detección de Anomalías)
* **¿Qué hace?** Actúa en "frío". Analiza la transacción de forma aislada y calcula qué tan "extraña" u "anómala" es la estructura matemática del movimiento basándose en montos y tarjetas, sin necesidad de conocer el historial previo.
* **Herramienta:** *Isolation Forest* (Enfoque No Supervisado).

### Capa 2: El Historial Temporal (Perfil de Comportamiento)
* **¿Qué hace?** El sistema no juzga la transacción sola, analiza el contexto del usuario. Calcula ráfagas de velocidad (¿cuántas compras lleva en la última hora?) y desviaciones financieras (¿este monto es normal para lo que acostumbra gastar este cliente?).
* **Herramienta:** *Ingeniería de Características Avanzada (Pandas y Transformaciones Relacionales)*.

### Capa 3: El Intérprete Inteligente (Enfoque Híbrido Neurosimbólico)
* **¿Qué hace?** Combina la potencia predictiva de un modelo avanzado de Machine Learning (que calcula la probabilidad matemática de fraude) con un **Motor de Reglas Lógicas de Negocio**. Cuando la Inteligencia Artificial duda en la "zona gris", las reglas duras inyectadas por expertos deciden si se bloquea el movimiento o se envía a una mesa de revisión manual, salvando a los clientes honestos.
* **Herramientas:** *LightGBM* (Clasificador) y *Lógica Condicional Dinámica*.

---

## 📈 Hallazgos e Impacto Real de Negocio

Evaluamos el sistema utilizando un conjunto masivo de datos transaccionales del mundo real históricos (**IEEE-CIS / Vesta Corporation**) que consta de casi 500,000 registros, validando el desempeño con transacciones simuladas del futuro.

Al activar el Motor de Reglas Simbólicas sobre la inteligencia artificial tradicional, obtuvimos los siguientes resultados de impacto operativo:

* **Protección Activa:** El sistema intercepta de forma automatizada transacciones fraudulentas críticas, mitigando el impacto financiero directo desde el primer segundo.
* **Reducción Masiva de Fricción:** Los modelos tradicionales bloqueaban por error a *20,098* clientes legítimos. Nuestra solución híbrida redujo esa cifra radicalmente a solo **7,575** casos sospechosos.

> 💡 **¡Evitamos 12,523 bloqueos erróneos a clientes honestos!** Esto representa una drástica reducción en la pérdida de clientes por malas experiencias de usuario y alivia de forma masiva la carga operativa en el centro de atención telefónica o equipos de soporte.

---

## 💻 Herramientas y Tecnologías Utilizadas

Para garantizar la estabilidad y la velocidad de procesamiento del proyecto, se seleccionaron tecnologías estándares de la industria tecnológica:

* **Python 3.12:** Lenguaje principal de desarrollo.
* **Pandas y NumPy:** Para la ingesta masiva de datos estructurados, optimización de memoria RAM y cálculo de perfiles de comportamiento transaccional.
* **Scikit-Learn:** Implementación de la capa estadística no supervisada para el aislamiento de anomalías.
* **LightGBM (Light Gradient Boosting Machine):** Algoritmo de alta velocidad optimizado por Microsoft, seleccionado específicamente por su bajo consumo de recursos de cómputo y eficiencia con grandes volúmenes de datos transaccionales.
* **KaggleHub API:** Para la automatización del proceso de descarga e ingesta segura de fuentes de datos crudas de la competencia oficial IEEE-CIS.

---

## 📁 Estructura del Repositorio

El proyecto se estructuró siguiendo las mejores prácticas de modularidad de la ingeniería de datos:

* `data/raw/`: Almacenamiento seguro de las fuentes de datos originales y crudas de transacciones e identidad.
* `data/processed/`: Repositorio intermedio para los datos limpios y enriquecidos con los scores de anomalía.
* `fraude.py` / `etl_pipeline.py`: Fase de Extracción y unión relacional masiva de datos.
* `transform_avanzado.py`: Fase de Transformación y creación de variables de comportamiento temporal de tarjetas.
* `entrenar_modelo.py`: Configuración y evaluación del motor probabilístico principal de Machine Learning.
* `sistema_neurosimbolico.py`: Orquestador híbrido final que ejecuta el motor de reglas lógicas de negocio para refinar la precisión del sistema.