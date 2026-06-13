# Usamos una imagen ligera oficial de Python basada en Debian-Slim para producción
FROM python:3.12-slim

# Establecer variables de entorno para optimizar Python dentro del contenedor
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crear y establecer el directorio de trabajo corporativo
WORKDIR /app

# Instalar dependencias del sistema necesarias para la ejecución de LightGBM (libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar UV de forma global dentro del contenedor para acelerar el despliegue
COPY --from=astral-sh/uv:latest /uv /uvx /bin/

# Copiar el archivo de dependencias congeladas generado con uv
COPY requirements.txt .

# Instalar los paquetes directamente en el sistema del contenedor usando UV de forma ultrarrápida
RUN uv pip install --system -r requirements.txt

# Copiar el resto del código fuente del pipeline neuro-simbólico al contenedor
COPY . .

# Exponer el puerto estándar por si en el futuro levantas una API de inferencia (FastAPI) o MLflow
EXPOSE 5000

# Comando por defecto para asegurar que la lógica de validación interna opera sin fallos
CMD ["python", "test_pipeline.py"]