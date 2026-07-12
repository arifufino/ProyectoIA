# Progreso de EduFeel IA

## Estado

- [x] Estructura modular del proyecto
- [x] Dataset CSV balanceado: 50 positivos, 50 neutrales y 50 negativos
- [x] Limpieza y validación de datos en español
- [x] TF-IDF con Regresión Logística y Naive Bayes configurables
- [x] Métricas completas y persistencia Joblib
- [x] API FastAPI con seis endpoints
- [x] Historial SQLite
- [x] Interfaz Streamlit con seis secciones
- [x] Manejo de errores solicitado
- [x] Suite Pytest
- [x] Documentación técnica, de usuario, informe y guiones
- [x] Verificación final ejecutada en el entorno
- [x] Commit y push final

## Notas

Los artefactos generados (`.joblib` y `.db`) se excluyen del repositorio y se crean al entrenar/usar la aplicación.

Verificación del 12 de julio de 2026: 15 pruebas aprobadas; FastAPI y Streamlit iniciaron correctamente; entrenamiento predeterminado con 120 registros y prueba con 30; los tres comentarios de aceptación se clasificaron como positivo, neutral y negativo y se persistieron en SQLite.
