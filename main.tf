# 1. Definición de Providers (AWS)
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1" # Región estándar de AWS
}

# 2. Variable para modularizar el nombre del proyecto
variable "project_name" {
  type    = string
  default = "prevencion-fraude-neurosimbolico"
}

# 3. Data Lake Seguro (S3) para almacenar Datasets y Artefactos de MLflow
resource "aws_s3_bucket" "fraud_data_lake" {
  bucket        = "gustavo-maldonado-${var.project_name}-datalake"
  force_destroy = false # Evita que se borren datos por accidente en producción

  tags = {
    Name        = "Fraud Detection Data Lake"
    Environment = "Production"
    ManagedBy   = "Terraform"
  }
}

# 4. Blindaje de Seguridad: Bloqueo explícito de acceso público al Bucket S3
resource "aws_s3_bucket_public_access_block" "fraud_s3_security_block" {
  bucket = aws_s3_bucket.fraud_data_lake.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# 5. Cifrado en Reposo (Server-Side Encryption) para cumplimiento regulatorio financiero
resource "aws_s3_bucket_server_side_encryption_configuration" "fraud_s3_encryption" {
  bucket = aws_s3_bucket.fraud_data_lake.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}