# 📊 Hallazgos del Proyecto: Prevención de Fraude Híbrida / Project Findings: Hybrid Fraud Prevention

Este documento explica de manera clara y accesible el problema de negocio que estamos resolviendo, cómo logramos solucionarlo y por qué nuestra metodología es altamente efectiva.

This document explains in a clear and accessible manner the business problem we are solving, how we achieved it, and why our methodology is highly effective.

---

## 🇪🇸 Versión en Español

### 1. ¿Qué problema estamos solucionando?
En el comercio electrónico y las plataformas de pago, el enemigo número uno de las ventas legítimas no es solo el fraude en sí, sino los **Falsos Positivos**. Un falso positivo ocurre cuando nuestro sistema de seguridad confunde a un cliente real con un criminal, bloqueando su tarjeta de forma errónea. 

Esto provoca tres problemas graves:
* **Pérdida de ventas:** El cliente se frustra, abandona el carrito de compras y se va con la competencia.
* **Saturación en soporte:** Los clientes llaman molestos al equipo de atención y reclamaciones, elevando los costos operativos.
* **Fricción con la marca:** Se daña la reputación de la plataforma de pagos.

**Nuestro objetivo principal:** Reducir drásticamente estos bloqueos erróneos sin dejar pasar el fraude real.

### 2. ¿Cómo lo logramos?
En lugar de depender de un solo modelo matemático aislado, construimos un **Ecosistema de Defensa en 3 Capas** que analiza el comportamiento del usuario desde diferentes ángulos:

* **Capa 1: El Radar de Rarezas (Isolation Forest):** Escanea millones de datos buscando comportamientos "extraños" en general (por ejemplo, transacciones en horarios inusuales o montos atípicos).
* **Capa 2: Inteligencia Predictiva (LightGBM):** Calcula con precisión matemática la probabilidad de que una transacción sea fraudulenta basándose en patrones históricos de fraude.
* **Capa 3: El Intérprete Experto (Motor de Reglas Simbólicas):** Actúa como un comité de analistas humanos automatizado en milisegundos. Cuando el modelo predictivo tiene dudas (en la "Zona Gris"), esta capa aplica reglas lógicas duras (ej. *"Si el monto es 3 veces más alto de lo normal para esta tarjeta Y además ocultó la información de su dispositivo móvil, bloquéalo"*).

### 3. ¿Por qué funcionó este método? (Impacto de Negocio)
La combinación de inteligencia artificial predictiva con lógica de negocio humana nos dio resultados extraordinarios en nuestro último grupo de prueba (**118,108 transacciones evaluadas**):

* **Falsos Positivos Anteriores:** 20,256 bloqueos erróneos.
* **Falsos Positivos Actuales:** 7,572 bloqueos erróneos.
* **Reducción de Fricción:** **¡Se evitaron 12,684 bloqueos injustificados a clientes legítimos!**

**¿Por qué es superior?** Los modelos de IA convencionales suelen ser "cajas negras" difíciles de ajustar y tienden a volverse demasiado estrictos, castigando al cliente honesto. Al inyectar un motor de reglas, le damos al sistema la capacidad de aplicar el "sentido común" corporativo sobre los casos dudosos, logrando un balance perfecto entre seguridad y fluidez comercial.

---

## 🇺🇸 English Version

### 1. What problem are we solving?
In e-commerce and payment platforms, the number one enemy of legitimate sales is not just fraud itself, but **False Positives**. A false positive occurs when our security system mistakes an innocent, real customer for a criminal, erroneously declining their card.

This triggers three severe business problems:
* **Lost Sales:** Frustrated customers abandon their shopping carts and switch to competitors.
* **Support Saturation:** Angry customers flood the customer support and claims teams, driving up operational costs.
* **Brand Friction:** The payment platform's reputation suffers long-term damage.

**Our Core Objective:** Drastically reduce these erroneous blocks without letting real fraud slip through.

### 2. How did we achieve it?
Instead of relying on a single, isolated mathematical model, we built a **3-Layer Defense Ecosystem** that analyzes user behavior from multiple perspectives:

* **Layer 1: The Oddity Radar (Isolation Forest):** Scans millions of data points looking for general "strange" behaviors (e.g., transactions at unusual hours or atypical amounts).
* **Layer 2: Predictive Intelligence (LightGBM):** Calculates with mathematical precision the probability of a transaction being fraudulent based on historical fraud patterns.
* **Layer 3: The Expert Interpreter (Symbolic Rules Engine):** Acts as a automated, millisecond-fast panel of human risk analysts. When the predictive model is uncertain (in the "Grey Zone"), this layer applies hard logical business rules (e.g., *"If the transaction amount is 3 times higher than normal for this card AND they hid their mobile device fingerprint, block it"*).

### 3. Why did this method work? (Business Impact)
Combining predictive Artificial Intelligence with human business logic yielded extraordinary results in our latest test batch (**118,108 evaluated transactions**):

* **Previous False Positives:** 20,256 erroneous declines.
* **Current False Positives:** 7,572 erroneous declines.
* **Friction Reduction:** **12,684 unjustified blocks to legitimate customers were successfully prevented!**

**Why is it superior?** Standard AI models often operate as "black boxes" that are difficult to tune and tend to become overly strict, punishing honest buyers. By injecting a rules engine, we empower the system to apply corporate "common sense" to borderline cases, achieving a flawless balance between robust security and seamless sales volume.