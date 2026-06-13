import os
import unittest
import yaml
import numpy as np
import pandas as pd

# Intentamos importar las funciones del pipeline para testearlas
from generar_eda import cargar_configuracion

class TestPipelineFraude(unittest.TestCase):

    def setUp(self):
        """Configuración de un entorno de prueba controlado."""
        self.ruta_config_test = "config_test_temp.yaml"
        self.mock_config = {
            'data': {
                'processed_dir': 'data/processed',
                'raw_dir': 'data/raw'
            },
            'isolation_forest': {'contamination': 0.04, 'n_estimators': 100, 'random_state': 42},
            'lightgbm': {'learning_rate': 0.05, 'n_estimators': 200, 'num_leaves': 31, 'random_state': 42},
            'rules': {'high_risk_amount': 5000.0}
        }
        # Crear un archivo de configuración temporal para las pruebas
        with open(self.ruta_config_test, 'w', encoding='utf-8') as f:
            yaml.dump(self.mock_config, f)

    def tearDown(self):
        """Limpieza del entorno de pruebas tras la ejecución."""
        if os.path.exists(self.ruta_config_test):
            os.remove(self.ruta_config_test)

    def test_cargar_configuracion_exitosa(self):
        """Validar que el cargador YAML procese correctamente la estructura de datos."""
        config = cargar_configuracion(self.ruta_config_test)
        self.assertEqual(config['isolation_forest']['n_estimators'], 100)
        self.assertEqual(config['rules']['high_risk_amount'], 5000.0)
        self.assertIn('lightgbm', config)

    def test_motor_reglas_simbolicas_mock(self):
        """Simula y valida el comportamiento lógico duro de la Capa 3 ante montos críticos."""
        # Umbral configurado
        umbral_critico = self.mock_config['rules']['high_risk_amount']
        
        # Caso 1: Transacción normal que requiere evaluación probabilística (LightGBM)
        monto_seguro = 150.0
        requiere_bloqueo_inmediato = monto_seguro > umbral_critico
        self.assertFalse(requiere_bloqueo_inmediato, "Una transacción baja no debe activar la regla dura.")
        
        # Caso 2: Transacción masiva que activa el Razonamiento Simbólico (Auditoría Mandatoria)
        monto_fraude = 7500.0
        requiere_bloqueo_inmediato = monto_fraude > umbral_critico
        self.assertTrue(requiere_bloqueo_inmediato, "Una transacción mayor al umbral debe encender alertas duras inmediatamente.")

if __name__ == "__main__":
    unittest.main()