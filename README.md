# Prevención de Fraude - Sistema Híbrido Neuro-Simbólico / Fraud Prevention - Neuro-Symbolic Hybrid System

Este repositorio contiene un pipeline de producción de nivel empresarial diseñado para la detección y mitigación de fraudes en transacciones de comercio electrónico utilizando el dataset masivo de **IEEE-CIS**. 

*Nota: La documentación técnica detallada sobre decisiones de diseño se encuentra en [TECNICO.md](TECNICO.md).*

---

## 🇪🇸 Versión en Español

La arquitectura adopta un enfoque **Neuro-Simbólico (Híbrido)** dividido en tres capas operacionales para resolver el problema del desbalance extremo de clases y minimizar los falsos positivos que afectan la experiencia del usuario en plataformas financieras.

### 🏗️ Arquitectura del Sistema
1. **Capa 1 (Estadística - Isolation Forest):** Radar espacial que analiza anomalías estructurales basadas en montos y volumen transaccional de tarjetas.
2. **Capa 2 (Probabilística - LightGBM):** Clasificador avanzado optimizado mediante inyección de dependencias y balanceo de carga dinámico (`scale_pos_weight`).
3. **Capa 3 (Simbólica - Motor de Reglas):** Lógica dura de negocio basada en conocimiento experto que actúa como última línea de defensa frente a falsos positivos masivos.

### 🚀 Stack Tecnológico y Gobernanza DevSecOps
* **Gestor de Entorno:** Optimizado con **`uv`** (Astral), reduciendo el tiempo de resolución e instalación de dependencias a milisegundos.
* **Gobernanza de Modelos:** Integración con **`MLflow`** para el rastreo automatizado de experimentos, parámetros, métricas de negocio (AUPRC, Matriz de Confusión) y registro del binario del modelo.
* **CI/CD Automático:** Pipeline de GitHub Actions (`devsecops_pipeline.yml`) que ejecuta escaneos estáticos de seguridad (**SAST con Bandit**) y suites de pruebas unitarias (`test_pipeline.py`) en cada push.
* **Infraestructura como Código:** Configuración de **`Terraform`** (`main.tf`) para instanciar y blindar un Data Lake en AWS S3 con cifrado en reposo AES256.
* **Portabilidad:** Contenedorización profesional optimizada mediante `Dockerfile` y `.dockerignore`.

### 🛠️ Ejecución Local

#### Prerrequisitos
Asegúrate de contar con `uv` instalado en tu sistema.

```powershell
# 1. Clonar e inicializar el entorno virtual con uv
uv venv
.venv\Scripts\Activate.ps1

# 2. Instalar dependencias congeladas
uv pip install -r requirements.txt

# 3. Ejecutar Análisis Exploratorio de Datos (EDA)
python generar_eda.py

# 4. Entrenar el Modelo y registrar en MLflow
python entrenar_modelo.py

# 5. Levantar la interfaz gráfica de MLflow
mlflow ui


🇺🇸 English Version
🏗️ System Architecture
The system adopts a Neuro-Symbolic (Hybrid) approach divided into three operational layers to address extreme class imbalance and minimize false positives that impact customer experience:

Layer 1 (Statistical - Isolation Forest): Spatial radar analyzing structural anomalies based on transaction amounts and card velocity metrics.

Layer 2 (Probabilistic - LightGBM): Advanced classifier optimized via dependency injection (config.yaml) and dynamic class weight balancing (scale_pos_weight).

Layer 3 (Symbolic - Rules Engine): Hardcoded business logic driven by domain expertise acting as the final line of defense to filter out massive false positives.

🚀 Technology Stack & DevSecOps Governance
Environment Manager: Powered by uv (Astral), reducing dependency resolution and installation times to milliseconds.

Model Governance: Integrated with MLflow for automated tracking of experiments, parameters, business metrics (AUPRC, Confusion Matrix), and model binary registration.

Automated CI/CD: GitHub Actions pipeline (devsecops_pipeline.yml) executing Static Application Security Testing (SAST via Bandit) and unit testing suites (unittest) on every push.

Infrastructure as Code: Terraform configuration (main.tf) to provision and secure a Data Lake in AWS S3 with AES256 server-side encryption.

Portability: Professional containerization optimized through a robust Dockerfile and .dockerignore.

🛠️ Local Execution
Prerequisites
Ensure you have uv installed globally on your system.

PowerShell
# 1. Clone and initialize the virtual environment with uv
uv venv
.venv\Scripts\Activate.ps1

# 2. Install pinned dependencies
uv pip install -r requirements.txt

# 3. Run Exploratory Data Analysis and Graphics
python generar_eda.py

# 4. Train the Model and log to MLflow
python entrenar_modelo.py

# 5. Launch MLflow Graphical User Interface
mlflow ui