# Documentación Técnica de Arquitectura / Technical Architecture Documentation ⚙️🧠

Este documento detalla las especificaciones técnicas, decisiones de diseño, ingeniería de características y el blindaje de seguridad (DevSecOps) implementados en el pipeline de prevención de fraude.

This document details the technical specifications, design decisions, feature engineering, and security hardening (DevSecOps) implemented within the fraud prevention pipeline.

---

## 🇪🇸 Sección en Español

### ⚙️ Inyección de Dependencias y Configuración Centralizada
Para evitar malas prácticas de acoplamiento rígido (*hardcoding*), el sistema completo se gobierna a través de un archivo centralizado `config.yaml`. Las funciones del pipeline consumen dinámicamente este archivo para ajustar:
* **Capa 1 (Isolation Forest):** Niveles de contaminación, número de estimadores y semilla aleatoria.
* **Capa 2 (LightGBM):** Hiperparámetros de aprendizaje (`learning_rate`), número de hojas (`num_leaves`), estimadores (`n_estimators`) y paralelismo (`n_jobs`).
* **Capa 3 (Motor de Reglas):** Umbrales duros de negocio para montos transaccionales críticos.
* **Estructura de Datos:** Rutas de directorios locales y de nube para mitigar fallas en el sistema de archivos.

### 📊 Hallazgos del Pipeline e Ingeniería de Datos
Durante el entrenamiento del clasificador con el dataset masivo de **IEEE-CIS**, se identificaron comportamientos operativos clave:
* **Desbalance Extremo y Falsos Positivos:** El LightGBM probabilístico puro tiende a generar un volumen elevado de falsos positivos (20,256 transacciones legítimas bloqueadas erróneamente) debido a la naturaleza asimétrica de los datos financieros.
* **Validación de Feature Engineering:** La importancia de variables determinó que la métrica creada **`card1_transaction_count`** es el segundo factor de decisión más influyente en el LightGBM (1,018 puntos de importancia). Esto justifica la inclusión de reglas deterministas (Capa 3) para filtrar la fricción operativa generada por modelos netamente estadísticos.

### 🛡️ Blindaje de Seguridad e Infraestructura (DevSecOps)
1. **Análisis SAST (Bandit):** Configurado en GitHub Actions para interceptar vulnerabilidades críticas (como la deserialización insegura con archivos `pickle` al guardar modelos o artefactos de MLflow).
2. **Pruebas Unitarias Automatizadas:** Implementadas con `unittest` para asegurar la resiliencia del cargador de configuración YAML y certificar que el motor de reglas lógicas bloquee transacciones que excedan los límites financieros paramétricos.
3. **Seguridad en la Nube (Terraform):** El script de Terraform fuerza el bloqueo de accesos públicos en Amazon S3 (`block_public_acls = true`) y habilita Server-Side Encryption (SSE-S3) por defecto para cumplir con regulaciones financieras internacionales (PCI-DSS / GDPR).

---

## 🇺🇸 English Section

### ⚙️ Dependency Injection & Centralized Configuration
To prevent hardcoding anti-patterns, the entire pipeline is governed through a centralized `config.yaml` file. System components dynamically ingest this file to configure:
* **Layer 1 (Isolation Forest):** Contamination levels, estimator counts, and random states.
* **Layer 2 (LightGBM):** Learning rate, number of leaves, estimators, and multi-threading execution parameters (`n_jobs`).
* **Layer 3 (Rules Engine):** Deterministic business logic thresholds for critical transaction amounts.
* **Data Infrastructure:** Local and cloud directory paths to prevent file system failures.

### 📊 Pipeline Insights & Data Engineering
During classifier training using the massive **IEEE-CIS** dataset, critical operational insights were discovered:
* **Extreme Imbalance & False Positives:** The pure probabilistic LightGBM tends to generate a high volume of false positives (20,256 legitimate transactions wrongly flagged) due to the highly asymmetric nature of financial data.
* **Feature Engineering Validation:** Feature importance analysis revealed that the custom engineered metric **`card1_transaction_count`** is the second most influential decision factor for LightGBM (1,018 importance points). This strongly validates the inclusion of deterministic rules (Layer 3) to filter out operational friction caused by purely statistical models.

### 🛡️ Security Hardening & Infrastructure (DevSecOps)
1. **SAST Analysis (Bandit):** Integrated into GitHub Actions to intercept critical vulnerabilities (such as insecure deserialization flaws with `pickle` formats when logging MLflow artifacts and models).
2. **Automated Unit Testing:** Built via `unittest` to secure the resilience of the YAML configuration parser and certify that the symbolic rules engine correctly blocks transactions exceeding parameterized financial bounds.
3. **Cloud Security (Terraform):** The Terraform infrastructure manifest enforces public access blocks on Amazon S3 buckets (`block_public_acls = true`) and enables Server-Side Encryption (SSE-S3) by default to comply with international financial regulations (PCI-DSS / GDPR).